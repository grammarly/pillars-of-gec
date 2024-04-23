# Ensemnble names to filenames mapping 

Below there is a mapping between ensemble names as in the paper and correspoding filenames with ensemnple predictions.

## Names mapping

<table>
  <tr>
    <th> Ensemble name </th>
    <th>File names</th>
  </tr>
  <tr>
    <th>majority-voting (best 7) </th>
    <th>ens_m7___*</th>
  </tr>
  <tr>
    <th>majority-voting (best 3) </th>
    <th>ens_m3___*</th>
  </tr>
  <tr>
    <th>GRECO-rank-w (best 7) </th>
    <th>ens_greco_on_m7___*</th>
  </tr>
  <tr>
    <th>GPT-4-rank-prompt-a (clust 3) </th>
    <th>gpt_rank_on_clust3___*</th>
  </tr>
  <tr>
    <th>AGGR-RANK [GPT-4-rank-a(clust 3), majority-voting(best 7)] </th>
    <th>aggr_of_gpt_rank_on_clust3_and_m7___*</th>
  </tr>
  <tr>
    <th>AGGR-RANK [GPT-4-rank-a(clust 3), GRECO-rank-w(best 7)] </th>
    <th>aggr_of_gpt_rank_on_clust3_and_greco_on_m7___*</th>
  </tr>
  <tr>
    <th>MAJORITY-VOTING[ majority-voting(best 7),  GRECO-rank-w(best 7) ]  </th>
    <th>ens_m8_with_greco___*</th>
  </tr>
  <tr>
    <th>MAJORITY-VOTING[ majority-voting(best 7),  GRECO-rank-w(best 7), GPT-4-rank-a(clust 3) + AGGR-RANK ]  </th>
    <th>ens_m9_with_greco_and_gpt___*</th>
  </tr>

</table>