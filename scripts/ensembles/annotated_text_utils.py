from collections import namedtuple
from enum import Enum
import re
import inspect
import difflib
from collections import namedtuple


# This is the value for representing annotations with missing suggestions.
# For example, given annotated text "He said to me {on Friday=>NO_SUGGESTIONS}
# he will work late." the words "on Friday" will be underlined as error,
# but no replacement will be suggested.
NO_SUGGESTIONS = "NO_SUGGESTIONS"
DEFAULT = object()


class OverlapError(ValueError):
    pass


class OnOverlap(str, Enum):
    ERROR = "error"
    OVERRIDE = "override"
    SAVE_OLD = "save_old"
    MERGE_STRICT = "merge_strict"  # Merge annotations that coincide exatly
    MERGE_EXPAND = "merge_expand"  # Merge annotations and expand their spans


def merge_strict(text, overlapping, new_ann):
    """On-overlap handler that merges two annotations that coincide in spans."""

    if len(overlapping) > 1:
        raise OverlapError(
            "Merge supports only 1-1 merges. "
            f"Call is done for {len(overlapping)}-1 merge."
        )

    existing = overlapping[0]
    if existing.start != new_ann.start or existing.end != new_ann.end:
        raise OverlapError(
            "Strict merge can be performed for annotations that "
            "share the span exactly."
        )

    suggestions = _unique_list(existing.suggestions + new_ann.suggestions)

    meta = {**existing.meta, **new_ann.meta}

    text.remove(existing)
    text.annotate(
        new_ann.start,
        new_ann.end,
        suggestions,
        meta=meta,
        on_overlap=OnOverlap.ERROR,
    )


def merge_expand(text, overlapping, new_ann):
    """On-overlap handler that merges two annotations possibly expanding spans."""

    if len(overlapping) > 1:
        raise OverlapError(
            "Merge supports only 1-1 merges. "
            f"Call is done for {len(overlapping)}-1 merge."
        )

    existing = overlapping[0]

    start = min(existing.start, new_ann.start)
    end = max(existing.end, new_ann.end)

    suggestions = []
    for annotation in (existing, new_ann):
        prefix = text._text[start : annotation.start]
        suffix = text._text[annotation.end : end]
        for sugg in annotation.suggestions:
            suggestions.append(prefix + sugg + suffix)

    suggestions = _unique_list(suggestions)

    meta = {**existing.meta, **new_ann.meta}

    text.remove(existing)
    text.annotate(
        start, end, suggestions, meta=meta, on_overlap=OnOverlap.ERROR
    )


def _unique_list(array):
    """Leave only unique elements in the list saving their order."""

    res = []
    for x in array:
        if x not in res:
            res.append(x)

    return res


class TokenAnnotation(
    namedtuple(
        "TokenAnnotation",
        ["start", "end", "source_text", "suggestions", "meta"],
    )
):
    """A single annotation in the list of tokens.

    Args:
        start: starting position in the original list of tokens.
        end: ending position in the original list of tokens.
        source_text: piece of the original tokens that is being corrected.
        suggestions: list of suggestions.
        meta (dict, optinal): additional data associated with the annotation.
    """

    def __new__(cls, start, end, source_text, suggestions, meta=DEFAULT):

        if meta is DEFAULT:
            meta = {}
        return super().__new__(cls, start, end, source_text, suggestions, meta)

    def __hash__(self):
        return hash(
            (
                self.start,
                self.end,
                self.source_text,
                tuple(self.suggestions),
                tuple(self.meta.items()),
            )
        )

    def __eq__(self, other):
        return (
            self.start == other.start
            and self.end == other.end
            and self.source_text == other.source_text
            and tuple(self.suggestions) == tuple(other.suggestions)
            and tuple(sorted(self.meta.items()))
            == tuple(sorted(other.meta.items()))
        )

    @property
    def top_suggestion(self):
        """Return the first suggestion or None if there are none."""

        return self.suggestions[0] if self.suggestions else None

    def to_str(self, *, with_meta=True):
        """Return a string representation of the annotation.

        Example:
            >>> ann = TokenAnnotation(0, 1, 'helo', ['hello', 'hola'])
            >>> ann.to_str()
            '{helo=>hello|hola}'

        """
        if self.suggestions:
            repl = "|".join(self.suggestions)
        else:
            repl = NO_SUGGESTIONS

        meta_text = self._format_meta() if with_meta else ""
        return "{%s=>%s%s}" % (self.source_text, repl, meta_text)

    def _format_meta(self):
        return "".join(":::{}={}".format(k, v) for k, v in self.meta.items())


def span_intersect(spans, begin, end):
    """Check if interval [begin, end) intersects with any of given spans.

    Args:
        spans: list of (begin, end) pairs.
        begin (int): starting position of the query interval.
        end (int): ending position of the query interval.

    Return:
        Index of a span that intersects with [begin, end),
            or -1 if no such span exists.
    """

    def strictly_inside(a, b, x, y):
        """Test that first segment is strictly inside second one."""
        return x < a <= b < y

    for index, (b, e) in enumerate(spans):
        overlap = max(0, min(end, e) - max(begin, b))
        if overlap:
            return index
        if strictly_inside(b, e, begin, end):
            return index
        if strictly_inside(begin, end, b, e):
            return index

    return -1


# For internal representation Element is used.
# Each element is a string if it's not part of any annotation, or object of
# Annotation class. All of those are basic pieces of the AnnotatedText that
# cannot be split into parts.
Element = namedtuple("Element", "start end item")

# When annotations are merged, we limit list of suggestions in size
MAX_SUGGESTIONS = 20


def from_annotated_text_with_spaces(ann_text):
    """Get tokens and annotations from AnnotatedText format.

    Converts AnnotatedText to tokens-level annotated AnnotatedTokens. If
    annotations are inside of the text then annotations will be expanded to
    token boundaries.

    N.B.
    1. Highlights (annotations with no suggestions) will be discarded if they
       are merged with another annotation.
       {well=>Well}{-=>NO_SUGGESTIONS}{defined=>defined.} will be converted to
       {well-defined=>Well-defined.}
       The only case when highlight is saved is when it's not expanded at all.

    2. Annotations with multiple suggestions are expanded combinatorically.
       E.g. {white=>wide|Wide}{=>-spread|spread} will be converted to
       {white=>wide-spread|widespread|Wide-spread|Widespread}
    """

    elements = prepare_elements(ann_text)
    # e.g. 'aaa b{bb=>cc} ddd'
    # is converted into
    # [e(0,3,'aaa'), e(4,5,'b'), e(5,7,A(5,7,src='bb',sugg=['cc']), e(8,11,'ddd')]

    tokens = []
    annotations = []
    i = 0
    while i < len(elements):

        # find sequence of elements which can be merged
        # e.g. the group in () here: 'aaa (bb{b=>cc}) ddd'
        j = i
        at_beginning = True
        while not is_the_end_of_merging_sequence(elements, j, at_beginning):
            j += 1
            at_beginning = False

        src, suggestions, meta = merge(elements[i : j + 1])
        src = src.split()
        suggestions = [s.split() for s in suggestions]
        suggestions = _unique_list(suggestions)

        # If src and trg are different then create an annotation object.
        has_annotation = len(suggestions) != 1 or src != suggestions[0]
        if has_annotation:
            start = len(tokens)
            e = start + len(src)
            source_text = " ".join(src)
            suggestions = [" ".join(s) for s in suggestions]
            annotation = TokenAnnotation(
                start, e, source_text, suggestions, meta
            )
            annotations.append(annotation)

        # Accumulate consumed tokens
        tokens += src
        i = j + 1

    return tokens, annotations


def merge(elements):
    """Merge list of elements into single `src`, `trg`, and `meta`."""

    assert len(elements) >= 1

    highlights_count = 0
    res_meta = {}
    res_source_text = ""
    res_suggestions = [""]

    for i in range(len(elements)):
        current_item = elements[i].item

        if isinstance(current_item, str):
            res_source_text += current_item
            res_suggestions = [s + current_item for s in res_suggestions]

        elif isinstance(current_item, Annotation):
            # Fulfill suggestions for highlight
            if not current_item.suggestions:
                curr_suggestions = [current_item.source_text]
                highlights_count += 1
            else:
                curr_suggestions = current_item.suggestions

            res_source_text += current_item.source_text

            # Expand suggestions by adding all gathered suggestions with all
            # possible suffixes from current annotation.
            new_suggestions = []
            for existing in res_suggestions:
                for suffix in curr_suggestions:
                    # Add suffix to the end of existing suggestion
                    new_sugg_to_add = existing + suffix
                    new_suggestions.append(new_sugg_to_add)

            res_suggestions = _unique_list(new_suggestions)[:MAX_SUGGESTIONS]

            res_meta.update(current_item.meta)

        else:
            raise ValueError(f"Unknown item type: {type(current_item)}")

    if len(elements) == highlights_count == 1:  # Save the only highlight
        res_suggestions = []

    return res_source_text, res_suggestions, res_meta


def prepare_elements(ann_text):
    """Returns list of Elements from AnnotatedText.

    e.g. input:
        'a{a=>x} b{bb=>}c{=>cc}a cc'
    returns:
        [
            E(0,1,'a'),
            E(1,2,A(1,2,src='a',suggs=['x'])
            E(3,4,'b'),
            E(4,6,A(4,6,src='bb',suggs=['']),
            E(6,7,'c'),
            E(7,7,A(0,6,src='',suggs=['cc']),
            E(7,8,'a')
            E(9,11,'cc')
        ]
    """

    original = ann_text.get_original_text()
    if len(original) == 0:
        return []

    # Determine which characters to include separately
    chars_free_from_annotations = [True] * len(original)
    bounds_free_from_annotations = [True] * (len(original) + 1)
    annotations = ann_text.get_annotations()
    for ann in annotations:
        ann_len = ann.end - ann.start

        not_free_chars = [False] * ann_len
        chars_free_from_annotations[ann.start : ann.end] = not_free_chars

        not_free_bounds = [False] * (ann_len + 1)
        bounds_free_from_annotations[ann.start : ann.end + 1] = not_free_bounds

    # Accumulate str token
    # and add it to elements
    elements = []
    s = 0
    while s < len(original):
        # Find the bounds of token.
        # Stop on Annotation, whitespace
        # or beginning of Annotation
        e = s
        while (
            e < len(original)
            and original[e] != " "
            and chars_free_from_annotations[e]
        ):
            e += 1
            if not bounds_free_from_annotations[e]:
                break

        # Add token to elements
        token_str = original[s:e]
        if token_str != "":
            elements.append(Element(s, e, original[s:e]))

        # Define start of next token
        # start assigns to next char
        # if next char is not whitespace and not Annotation
        start_from_next_char = not (
            e < len(original)
            and original[e] != " "
            and chars_free_from_annotations[e]
        )
        s = e + start_from_next_char

    # Add annotations to elements
    for a in annotations:
        elements.append(Element(a.start, a.end, a))

    return sorted(elements, key=lambda x: (x.start, x.end))


def is_the_end_of_merging_sequence(elements, i, at_beginning):
    """Checks whether i'th element is the end of merging sequence

    Args:
        at_beginning (bool): True if i'th element is the first in the sequence

    Returns:
        True if current (i'th) and next elements are separated by space
        False otherwise
    """

    if i + 1 >= len(elements):
        return True

    if elements[i].end < elements[i + 1].start:
        return True

    current_item = elements[i].item

    next_element = elements[i + 1]
    next_item = next_element.item

    assert isinstance(current_item, (Annotation, str))
    assert isinstance(next_item, (Annotation, str))

    if isinstance(current_item, str) and isinstance(next_item, str):
        raise Exception("Wrong input data", "String tokens are not separated")

    elif isinstance(current_item, Annotation) and isinstance(next_item, str):
        # Current item's source text
        # and suggestions end with spaces
        if texts_end_with_space([current_item.source_text]) and (
            texts_end_with_space(current_item.suggestions)
        ):
            return True

        # Current item is insertion
        # and suggestions end with spaces
        # and suggestions are not spaces
        elif (
            texts_are_empty([current_item.source_text])
            and texts_end_with_space(current_item.suggestions)
            and not texts_are_spaces(current_item.suggestions)
        ):
            return True

        # Current item is deletion
        # and source text ends with space
        # and current item is beginning of merging sequence
        elif (
            texts_end_with_space([current_item.source_text])
            and texts_are_empty(current_item.suggestions)
            and at_beginning
        ):
            return True

        else:
            return False

    elif isinstance(current_item, str) and isinstance(next_item, Annotation):
        # Next item is insertion
        # and suggestions start with spaces
        # and suggestions are not spaces
        if (
            texts_are_empty([next_item.source_text])
            and texts_start_with_space(next_item.suggestions)
            and not texts_are_spaces(next_item.suggestions)
        ):
            return True

        # Next item's source text starts with space
        # and suggestions start with space
        elif texts_start_with_space([next_item.source_text]) and (
            texts_start_with_space(next_item.suggestions)
        ):
            return True

        else:
            return False

    # fmt: off
    elif (
        isinstance(current_item, Annotation)
        and isinstance(next_item, Annotation)
    ):

        # Current item is not an insertion
        # and either suggestions of current item end with spaces
        # or suggestions of next item start with spaces
        if (
            len(next_item.suggestions)
            and len(current_item.suggestions)
            and not texts_are_empty([current_item.source_text])
            and (
                texts_end_with_space(current_item.suggestions)
                or texts_start_with_space(next_item.suggestions)
            )
        ):
            return True

        else:
            return False


def texts_end_with_space(texts):
    """Checks list of str elements if all of them end with space."""

    if isinstance(texts, list):
        return all(text.endswith(" ") for text in texts)
    else:
        raise ValueError(f"{type(texts)} is not list")


def texts_are_empty(texts):
    """Checks list of str elements if all of them are empty."""

    if isinstance(texts, list):
        return all(text == "" for text in texts)
    else:
        raise ValueError(f"{type(texts)} is not list")


def texts_start_with_space(texts):
    """Checks list of str elements if all of them start with space."""

    if isinstance(texts, list):
        return all(text.startswith(" ") for text in texts)
    else:
        raise ValueError(f"{type(texts)} is not list")


def texts_are_spaces(texts):
    """Checks list of str elements if all of them are spaces."""

    if isinstance(texts, list):
        return all(text == " " for text in texts)
    else:
        raise ValueError(f"{type(texts)} is not list")



class MutableText:
    """Represents text that can be modified."""

    def __init__(self, text):
        self._text = text
        self._edits = []

    def __str__(self):
        """Pretend to be a normal string."""
        return self.get_edited_text()

    def __repr__(self):
        return "<MutableText({})>".format(repr(str(self)))

    def replace(self, start, end, value):
        """Replace substring with a value.

        Example:
            >>> t = MutableText('the red fox')
            >>> t.replace(4, 7, 'brown')
            >>> t.get_edited_text()
            'the brown fox'

        """
        self._edits.append((start, end, value))  # TODO: keep _edits sorted?

    def apply_edits(self):
        """Applies all edits made so far."""

        self._text = self.get_edited_text()
        self._edits = []

    def get_source_text(self):
        """Return string without pending edits applied.

        Example:
            >>> t = MutableText('the red fox')
            >>> t.replace(4, 7, 'brown')
            >>> t.get_source_text()
            'the red fox'
        """
        return self._text

    def get_edited_text(self):
        """Return text with all corrections applied."""

        result = []
        i = 0
        t = self._text
        for begin, end, val in sorted(self._edits, key=lambda x: (x[0], x[1])):
            result.append(t[i:begin])
            result.append(val)
            i = end
        result.append(t[i:])
        return "".join(result)


class AnnotatedText:
    """Text representation that allows easy replacements and annotations.

    This class also supports parsing meta data from annotations.

    Example:
        >>> s = 'Hi {wold=>World|world:::type=OOV Spell:::status=ok}'
        >>> anns = AnnotatedText(s).get_annotations()
        >>> anns[0].suggestions
        ['World', 'world']
        >>> anns[0].meta
        {'type': 'OOV Spell', 'status': 'ok'}

    """

    ANNOTATION_PATTERN = re.compile(r"\{([^{]*)=>(.*?)(:::[^:][^}]*)?\}")

    def __init__(self, text: str) -> None:

        if not isinstance(text, str):
            raise ValueError(f"`text` must be string, not {type(text)}")

        original = self.ANNOTATION_PATTERN.sub(r"\1", text)
        self._annotations = self._parse(text)
        self._text = original

    def __str__(self):
        """Pretend to be a normal string."""
        return self.get_annotated_text()

    def __repr__(self):
        return "<AnnotatedText('{}')>".format(self.get_annotated_text())

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self._text != other._text:
            return False

        if len(self._annotations) != len(other._annotations):
            return False

        for ann in other._annotations:
            if ann != self.get_annotation_at(ann.start, ann.end):
                return False

        return True

    def annotate(
        self,
        start,
        end,
        correct_value,
        append=None,
        meta=None,
        on_overlap=OnOverlap.ERROR,
    ):

        """Annotate substring as being corrected.

        Args:
            start (int): starting position of the substring to annotate.
            end (int): ending position of the substring to annotate.
            correct_value (str, iterable, None):
                one or more correction suggestions.
            meta (dict, optional): any additional info associated with the
                annotation. Defaults to an empty dict.
            on_overlap (callable or OnOverlap): What to do when two annotations
                overlap. By default - throws OverlapError.
                If callable is passed, when conflict is found, the callable is
                responsible for resolving the conflict (in-place). Parameters
                that will be passed to the handler:
                 * annotated text itself
                 * list of existing annotations that overlap
                 * new annotation

        Example:
            >>> t = AnnotatedText('the red fox')
            >>> t.annotate(4, 7, ['brown', 'white'])
            >>> t.get_annotated_text()
            'the {red=>brown|white} fox'

        """
        if start > end:
            raise ValueError(
                f"Start positition {start} should not greater "
                f"than end position {end}"
            )

        if append is not None:
            raise DeprecationWarning(
                "`append` has been deprecated. " "Use `on_overlap` instead"
            )

        if meta is None:
            meta = dict()

        bad = self._text[start:end]

        if isinstance(correct_value, str):
            suggestions = [correct_value]
        elif correct_value is None:
            suggestions = []
        else:
            suggestions = list(correct_value)

        new_ann = Annotation(start, end, bad, suggestions, meta)
        overlapping = self._get_overlaps(start, end)
        if overlapping:
            if callable(on_overlap):
                on_overlap(self, overlapping, new_ann)
            elif on_overlap == OnOverlap.SAVE_OLD:
                pass
            elif on_overlap == OnOverlap.OVERRIDE:
                for ann in overlapping:
                    self.remove(ann)
                self._annotations.append(new_ann)
            elif on_overlap == OnOverlap.ERROR:
                raise OverlapError(
                    f"Overlap detected: positions ({start}, {end}) with "
                    f"{len(overlapping)} existing annotations."
                )
            elif on_overlap == OnOverlap.MERGE_STRICT:
                merge_strict(self, overlapping, new_ann)
            elif on_overlap == OnOverlap.MERGE_EXPAND:
                merge_expand(self, overlapping, new_ann)
            else:
                raise ValueError(f"Unknown on_overlap action: {on_overlap}")
        else:
            self._annotations.append(new_ann)

    def _get_overlaps(self, start, end):
        """Find all annotations that overlap with given range."""

        res = []
        for ann in self._annotations:
            if span_intersect([(ann.start, ann.end)], start, end) != -1:
                res.append(ann)
            elif start == end and ann.start == ann.end and start == ann.start:
                res.append(ann)

        return res

    def undo_edit_at(self, index):
        """Undo the last edit made at the given position."""
        for i, (start, end, val) in enumerate(reversed(self._edits)):
            if start == index:
                self._edits.pop(-i - 1)
                return
        raise IndexError()

    def get_annotations(self):
        """Return list of all annotations in the text."""

        return self._annotations

    def iter_annotations(self):
        """Iterate the annotations in the text.

        This differs from `get_annotations` in that you can safely modify
        current annotation during the iteration. Specifically, `remove` and
        `apply_correction` are allowed. Adding and modifying annotations other
        than the one being iterated is not yet well-defined!

        Example:
            >>> text = AnnotatedText('{1=>One} {2=>Two} {3=>Three}')
            >>> for i, ann in enumerate(text.iter_annotations()):
            ...     if i == 0:
            ...         text.apply_correction(ann)
            ...     else:
            ...         text.remove(ann)
            >>> text.get_annotated_text()
            'One 2 3'

        Yields:
            Annotation instances

        """

        n_anns = len(self._annotations)
        i = 0
        while i < n_anns:
            yield self._annotations[i]
            delta = len(self._annotations) - n_anns
            i += delta + 1
            n_anns = len(self._annotations)

    def get_annotation_at(self, start, end=None):
        """Return annotation at the given position or region.

        If only `start` is provided, return annotation that covers that
        source position.

        If both `start` and `end` are provided, return annotation
        that matches (start, end) span exactly.

        Return `None` if no annotation was found.
        """

        if end is None:
            for ann in self._annotations:
                if ann.start <= start < ann.end:
                    return ann
        else:
            for ann in self._annotations:
                if ann.start == start and ann.end == end:
                    return ann

        return None

    def _parse(self, text):
        """Return list of annotations found in the text."""

        anns = []
        amend = 0
        for match in self.ANNOTATION_PATTERN.finditer(text):
            source, suggestions, meta_text = match.groups()
            start = match.start() - amend
            end = start + len(source)
            if suggestions != NO_SUGGESTIONS:
                suggestions = suggestions.split("|")
            else:
                suggestions = []

            if meta_text:
                key_values = [
                    x.partition("=") for x in meta_text.split(":::")[1:]
                ]
                meta = {k: v for k, _, v in key_values}
            else:
                meta = {}

            ann = Annotation(
                start=start,
                end=end,
                source_text=source,
                suggestions=suggestions,
                meta=meta,
            )
            anns.append(ann)
            amend += match.end() - match.start() - len(source)

        return anns

    def remove(self, annotation):
        """Remove annotation, replacing it with the original text."""

        try:
            self._annotations.remove(annotation)
        except ValueError:
            raise ValueError("{} is not in the list".format(annotation))

    def apply_correction(self, annotation, level=0):
        """Remove annotation, replacing it with the corrected text.

        Example:
            >>> text = AnnotatedText('{one=>ONE} {too=>two}')
            >>> a = text.get_annotations()[0]
            >>> text.apply_correction(a)
            >>> text.get_annotated_text()
            'ONE {too=>two}'
        """

        try:
            self._annotations.remove(annotation)
        except ValueError:
            raise ValueError("{} is not in the list".format(annotation))

        text = MutableText(self._text)
        if annotation.suggestions:
            repl = annotation.suggestions[level]
        else:
            repl = annotation.source_text  # for NO_SUGGESTIONS annotations
        text.replace(annotation.start, annotation.end, repl)
        self._text = text.get_edited_text()

        # Adjust other annotations
        delta = len(repl) - len(annotation.source_text)
        for i, a in enumerate(self._annotations):
            if a.start >= annotation.start:
                a = a._replace(start=a.start + delta, end=a.end + delta)
                self._annotations[i] = a

    def get_original_text(self):
        """Return the original (unannotated) text.

        Example:
            >>> text = AnnotatedText('{helo=>Hello} world!')
            >>> text.get_original_text()
            'helo world!'
        """

        return self._text

    def get_corrected_text(self, level=0):
        """Return the unannotated text with all corrections applied.

        Example:
            >>> text = AnnotatedText('{helo=>Hello} world!')
            >>> text.get_corrected_text()
            'Hello world!'
        """

        text = MutableText(self._text)
        for ann in self._annotations:
            try:
                text.replace(ann.start, ann.end, ann.suggestions[level])
            except IndexError:
                pass

        return text.get_edited_text()

    def get_annotated_text(self, *, with_meta=True):
        """Return the annotated text.

        Example:
            >>> text = AnnotatedText('helo world!')
            >>> text.annotate(0, 4, 'Hello', meta={'key': 'value'})
            >>> text.get_annotated_text()
            '{helo=>Hello:::key=value} world!'
            >>> text.get_annotated_text(with_meta=False)
            '{helo=>Hello} world!'

        Args:
            with_meta: Whether to serialize `meta` fields.

        Returns:
            str
        """

        text = MutableText(self._text)
        for ann in self._annotations:
            text.replace(ann.start, ann.end, ann.to_str(with_meta=with_meta))

        return text.get_edited_text()

    def combine(self, other, discard_overlap=True):
        """Combine annotations with other text's annotations.

        Args:
            other (AnnotatedText): Other text with annotations.
            discard_overlap (bool): If `False`, will raise an error when two
                annotations overlap, otherwise silently discards them, giving
                priority to current object's annotations.
        """

        if not isinstance(other, AnnotatedText):
            raise ValueError(
                "Expected `other` to be {}, received {}".format(
                    AnnotatedText, type(other)
                )
            )

        if self.get_original_text() != other.get_original_text():
            raise ValueError(
                "Cannot combine with text from different " "original text"
            )

        on_overlap = "save_old" if discard_overlap else "error"
        for ann in other.get_annotations():
            self.annotate(
                ann.start,
                ann.end,
                ann.suggestions,
                meta=ann.meta,
                on_overlap=on_overlap,
            )

    @staticmethod
    def join(join_token, ann_texts):
        """Joins annotated texts by join_token.

        It's an analogy for `join_token.join(ann_texts)` but for AnnotatedText
        class.

        Args:
            join_token (str): Token to use for joining.
            ann_texts (list[AnnotatedText]): AnnotatedTexts to join.

        Returns:
            AnnotatedText
        """

        for ann_text in ann_texts:
            if not isinstance(ann_text, AnnotatedText):
                raise ValueError(
                    f"{str(ann_text)} is not of class AnnotatedText"
                )

        s = join_token.join(str(a) for a in ann_texts)

        return AnnotatedText(s)


class Annotation(
    namedtuple(
        "Annotation", ["start", "end", "source_text", "suggestions", "meta"]
    )
):
    """A single annotation in the text.

    Args:
        start: starting position in the original text.
        end: ending position in the original text.
        source_text: piece of the original text that is being corrected.
        suggestions: list of suggestions.
        meta (dict, optinal): additional data associated with the annotation.

    """

    def __new__(cls, start, end, source_text, suggestions, meta=DEFAULT):

        if meta is DEFAULT:
            meta = {}
        return super().__new__(cls, start, end, source_text, suggestions, meta)

    def __hash__(self):
        return hash(
            (
                self.start,
                self.end,
                self.source_text,
                tuple(self.suggestions),
                tuple(self.meta.items()),
            )
        )

    def __eq__(self, other):
        return (
            self.start == other.start
            and self.end == other.end
            and self.source_text == other.source_text
            and tuple(self.suggestions) == tuple(other.suggestions)
            and tuple(sorted(self.meta.items()))
            == tuple(sorted(other.meta.items()))
        )

    @property
    def top_suggestion(self):
        """Return the first suggestion or None if there are none."""

        return self.suggestions[0] if self.suggestions else None

    def to_str(self, *, with_meta=True):
        """Return a string representation of the annotation.

        Example:
            >>> ann = Annotation(0, 4, 'helo', ['hello', 'hola'])
            >>> ann.to_str()
            '{helo=>hello|hola}'

        """
        if self.suggestions:
            repl = "|".join(self.suggestions)
        else:
            repl = NO_SUGGESTIONS

        meta_text = self._format_meta() if with_meta else ""
        return "{%s=>%s%s}" % (self.source_text, repl, meta_text)

    def _format_meta(self):
        return "".join(":::{}={}".format(k, v) for k, v in self.meta.items())


class MutableTokens:
    """Represents list of tokens that can be modified."""

    def __init__(self, tokens):
        if isinstance(tokens, str):
            tokens = tokens.strip().split(" ")
        self._tokens = tokens
        self._edits = []

    def __str__(self):
        """Pretend to be a normal string."""
        return self.get_edited_text()

    def __repr__(self):
        return "<MutableTokens({})>".format(repr(str(self)))

    def replace(self, start, end, value):
        """Replace sublist with a value.

        Example:
            >>> t = MutableTokens('the red fox')
            >>> t.replace(1, 2, 'brown')
            >>> t.get_edited_text()
            'the brown fox'
        """
        self._edits.append((start, end, value))  # TODO: keep _edits sorted?

    def apply_edits(self):
        """Applies all edits made so far."""
        self._tokens = self.get_edited_tokens()
        self._edits = []

    def get_source_tokens(self):
        """Return list of tokens without pending edits applied.

        Example:
            >>> t = MutableTokens('the red fox')
            >>> t.replace(1, 2, 'brown')
            >>> t.get_source_tokens()
            ['the', 'red', 'fox']
        """
        return self._tokens

    def get_source_text(self):
        """Return string without no pending edits applied.

        Example:
            >>> t = MutableTokens('the red fox')
            >>> t.replace(1, 2, 'brown')
            >>> t.get_source_text()
            'the red fox'
        """
        return " ".join(self.get_source_tokens())

    def get_edited_tokens(self, *, highlight=False):
        """Return tokens with all corrections applied.

        Args:
            highlight (bool): If True, keep NO_SUGGESTIONS markup in correction.
                This signals the error is highlighted but no suggestion was
                provided.
        """
        result = []
        i = 0
        t = self._tokens
        for begin, end, val in sorted(self._edits, key=lambda x: (x[0], x[1])):
            result.extend(t[i:begin])
            if not highlight and "NO_SUGGESTIONS" in val:
                result.extend(t[begin:end])
            elif val:
                result.extend(val.split(" "))
            i = end
        result.extend(t[i:])
        return result

    def get_edited_text(self, *, highlight=False):
        """Return text with all corrections applied."""
        return " ".join(self.get_edited_tokens(highlight=highlight))


class AnnotatedTokens:
    """Tokens representation that allows easy replacements and annotations.

    All text representations is made by joining tokens by space. This format of
    converting is assumed by default in this class.
    """

    def __init__(self, tokens):
        """Creates object.

        Args:
            tokens (AnnotatedText or str or list): Allowed formats:
                - AnnotatedText instance
                - a string of tokens joined by space
                - a list of tokens.
        """

        self._annotations = []
        if isinstance(tokens, AnnotatedText):
            (
                self._tokens,
                self._annotations,
            ) = from_annotated_text_with_spaces(tokens)
        elif isinstance(tokens, str):
            self._tokens = tokens.split(" ")
        else:
            self._tokens = tokens

    def __str__(self):
        return self.get_annotated_text()

    def __repr__(self):
        return "<AnnotatedTokens('{}')>".format(self.get_annotated_text())

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        if self._tokens != other._tokens:
            return False

        if len(self._annotations) != len(other._annotations):
            return False

        for ann in other._annotations:
            if ann != self.get_annotation_at(ann.start, ann.end):
                return False

        return True

    def annotate(
        self, start, end, correct_value, meta=None, on_overlap=OnOverlap.ERROR
    ):
        """Annotate sublist as being corrected.

        Args:
            start (int): starting position of the sublist to annotate.
            end (int): ending position of the sublist to annotate.
            correct_value (str, iterable, None): one or more correction
                suggestions, each being a token list joined by space.
            meta (dict, optional): any additional info associated with the
                annotation. Defaults to an empty dict.

        Example:
            >>> t = AnnotatedTokens('the red fox')
            >>> t.annotate(1, 2, ['brown', 'white'])
            >>> t.get_annotated_text()
            'the {red=>brown|white} fox'
        """

        if start > end:
            raise ValueError(
                f"Start positition {start} should not greater "
                f"than end position {end}"
            )

        if meta is None:
            meta = dict()

        bad = " ".join(self._tokens[start:end])

        if isinstance(correct_value, str):
            suggestions = [correct_value]
        elif correct_value is None:
            suggestions = []
        else:
            suggestions = list(correct_value)

        new_ann = TokenAnnotation(start, end, bad, suggestions, meta)
        overlapping = self._get_overlaps(start, end)
        if overlapping:
            if callable(on_overlap):
                on_overlap(self, overlapping, new_ann)
            elif on_overlap == OnOverlap.SAVE_OLD:
                pass
            elif on_overlap == OnOverlap.OVERRIDE:
                for ann in overlapping:
                    self.remove(ann)
                self._annotations.append(new_ann)
            elif on_overlap == OnOverlap.ERROR:
                raise OverlapError(
                    f"Overlap detected: positions ({start}, {end}) with "
                    f"{len(overlapping)} existing annotations."
                )
            elif on_overlap == OnOverlap.MERGE_STRICT:
                merge_strict(self, overlapping, new_ann)
            elif on_overlap == OnOverlap.MERGE_EXPAND:
                merge_expand(self, overlapping, new_ann)
            else:
                raise ValueError(f"Unknown on_overlap action: {on_overlap}")
        else:
            self._annotations.append(new_ann)

    def _get_overlaps(self, start, end):
        """Find all annotations that overlap with given range."""

        res = []
        for ann in self._annotations:
            if span_intersect([(ann.start, ann.end)], start, end) != -1:
                res.append(ann)
            elif start == end and ann.start == ann.end and start == ann.start:
                res.append(ann)

        return res

    def get_annotations(self):
        """Return list of all annotations in the text."""
        return self._annotations

    def iter_annotations(self):
        """Iterate the annotations in the text.

        This differs from `get_annotations` in that you can safely modify
        current annotation during the iteration. Specifically, `remove` and
        `apply_correction` are allowed. Adding and modifying annotations other
        than the one being iterated is not yet well-defined!

        Example:
            >>> tokens = AnnotatedTokens('1 2 3')
            >>> tokens.annotate(0, 1, 'One')
            >>> tokens.annotate(1, 2, 'Two')
            >>> tokens.annotate(2, 3, 'Three')
            >>> for i, ann in enumerate(tokens.iter_annotations()):
            ...     if i == 0:
            ...         tokens.apply_correction(ann)
            ...     else:
            ...         tokens.remove(ann)
            >>> tokens.get_annotated_text()
            'One 2 3'

        Yields:
            TokenAnnotation instances.
        """

        n_anns = len(self._annotations)
        i = 0
        while i < n_anns:
            yield self._annotations[i]
            delta = len(self._annotations) - n_anns
            i += delta + 1
            n_anns = len(self._annotations)

    def get_annotation_at(self, start, end):
        """Return annotation for the region (start, end) or None."""

        for ann in self._annotations:
            if ann.start == start and ann.end == end:
                return ann

    def remove(self, annotation):
        """Remove annotation, replacing it with the original text."""

        try:
            self._annotations.remove(annotation)
        except ValueError:
            raise ValueError("{} is not in the list".format(annotation))

    def filter_annotations(self, f=None):
        """Filter annotations using function passed as an argument.

        Args:
            f: function receiving the annotation as argument and optionally
             the AnnotatedTokens object itself. If the function returns False,
             remove annotation.
        """

        f_param = inspect.signature(f).parameters.values() if f else []
        if len(f_param) not in [1, 2]:
            raise ValueError(
                "Filter function only accepts 1 or 2 arguments."
                "Arguments received: {}".format(f_param)
            )
        for ann in self.iter_annotations():
            if len(f_param) == 2:
                result = f(ann, self)
            else:
                result = f(ann)
            if not result:
                self.remove(ann)

    def apply_correction(self, annotation, level=0):
        """Remove annotation, replacing it with the corrected text.

        Example:
            >>> tokens = AnnotatedTokens('one too')
            >>> tokens.annotate(0, 1, 'ONE')
            >>> tokens.annotate(1, 2, 'two')
            >>> a = tokens.get_annotations()[0]
            >>> tokens.apply_correction(a)
            >>> tokens.get_annotated_text()
            'ONE {too=>two}'
        """

        try:
            self._annotations.remove(annotation)
        except ValueError:
            raise ValueError("{} is not in the list".format(annotation))

        tokens = MutableTokens(self._tokens)
        if annotation.suggestions:
            repl = annotation.suggestions[level]
        else:
            repl = annotation.source_text  # for NO_SUGGESTIONS annotations
        tokens.replace(annotation.start, annotation.end, repl)
        self._tokens = tokens.get_edited_tokens()

        # Adjust other annotations
        source_text = annotation.source_text
        old_len = len(source_text.split(" ")) if source_text else 0
        new_len = len(repl.split(" ")) if repl else 0
        delta = new_len - old_len
        for i, a in enumerate(self._annotations):
            if a.start >= annotation.start:
                a = a._replace(start=a.start + delta, end=a.end + delta)
                self._annotations[i] = a

    def get_original_tokens(self):
        """Return the original (unannotated) tokens.

        Example:
            >>> tokens = AnnotatedTokens('helo world !')
            >>> tokens.annotate(0, 1, 'Hello')
            >>> tokens.get_original_tokens()
            ['helo', 'world', '!']
        """

        return self._tokens

    def get_original_text(self):
        """Return the original (unannotated) text.

        Example:
            >>> tokens = AnnotatedTokens('helo world !')
            >>> tokens.annotate(0, 1, 'Hello')
            >>> tokens.get_original_text()
            'helo world !'
        """

        return " ".join(self.get_original_tokens())

    def get_corrected_tokens(self, level=0):
        """Return the unannotated tokens with all corrections applied.

        Example:
            >>> tokens = AnnotatedTokens('helo world !')
            >>> tokens.annotate(0, 1, 'Hello')
            >>> tokens.get_corrected_tokens()
            ['Hello', 'world', '!']
        """

        tokens = MutableTokens(self._tokens)
        for ann in self._annotations:
            try:
                tokens.replace(ann.start, ann.end, ann.suggestions[level])
            except IndexError:
                pass

        return tokens.get_edited_tokens()

    def get_corrected_text(self, level=0):
        """Return the corrected (unannotated) text.

        Example:
            >>> tokens = AnnotatedTokens('helo world !')
            >>> tokens.annotate(0, 1, 'Hello')
            >>> tokens.get_corrected_text()
            'Hello world !'
        """

        return " ".join(self.get_corrected_tokens(level))

    def get_annotated_text(self, *, with_meta=True):
        """Return the annotated tokens in text format.

        Example:
            >>> tokens = AnnotatedTokens('helo . world!')
            >>> tokens.annotate(0, 2, 'Hello ,', meta={'key': 'value'})
            >>> tokens.get_annotated_text(with_meta=False)
            '{helo .=>Hello ,} world!'
            >>> tokens.get_annotated_text()
            '{helo .=>Hello ,:::key=value} world!'
        """

        tokens = MutableTokens(self._tokens)
        for ann in self._annotations:
            tokens.replace(ann.start, ann.end, ann.to_str(with_meta=with_meta))

        return " ".join(tokens.get_edited_tokens(highlight=True))

    def combine(self, other, discard_overlap=True):
        """Combine annotations with other text's annotations.

        Args:
            other (AnnotatedTokens): Other text with annotations.
            discard_overlap (bool): If `False`, will raise an error when two
                annotations overlap, otherwise silently discards them, giving
                priority to current object's annotations.
        """

        if not isinstance(other, AnnotatedTokens):
            raise ValueError(
                "Expected `other` to be {}, received {}".format(
                    AnnotatedTokens, type(other)
                )
            )

        if self.get_original_text() != other.get_original_text():
            raise ValueError(
                "Cannot combine with text from different " "original text"
            )

        on_overlap = "save_old" if discard_overlap else "error"
        for ann in other.get_annotations():
            self.annotate(
                ann.start,
                ann.end,
                ann.suggestions,
                meta=ann.meta,
                on_overlap=on_overlap,
            )

    def to_annotated_text_with_spaces(self):
        """Convert to AnnotatedText format, remaining spaces between tokens."""
        return AnnotatedText(self.get_annotated_text())


def align(source, target):
    """Create AnnotatedTokens object based on two sentences.

    Args:
        source (str or list<str>): Original sentence or tokens.
        target (str or list<str>): Corrected sentence or tokens.

    Return:
        AnnotatedTokens object.

    Example:
        >>> align("Hello world", "Hey world")
        <AnnotatedTokens('{Hello=>Hey} world')>
    """

    if isinstance(source, str):
        source = source.split()

    ann_tokens = AnnotatedTokens(source)
    for diff in _gen_diffs(source, target):
        l, r, repl = diff
        ann_tokens.annotate(l, r, repl)
    return ann_tokens


def _gen_diffs(original, translation, merge=True):
    tokens = _get_tokens(original)
    translation_tokens = _get_tokens(translation)

    matcher = difflib.SequenceMatcher(None, tokens, translation_tokens)
    diffs = list(matcher.get_opcodes())

    for diff in diffs:
        if _tag(diff) == "equal":
            continue

        _, i1, i2, j1, j2 = diff
        yield i1, i2, " ".join(translation_tokens[j1:j2])


def _get_tokens(str_or_list):
    if isinstance(str_or_list, str):
        return str_or_list.split()

    if isinstance(str_or_list, list):
        return str_or_list[:]

    raise ValueError("Cannot cast {} to list of tokens.".format(type(str_or_list)))


def _tag(diff):
    return diff[0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()

