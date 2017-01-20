import pandas as pd
import cPickle as pickle

from evaluation import p_and_e_combinations
from evaluation import precision_at_k
from evaluation import discounted_cumulative_gain_at_k
from evaluation import rank_biased_precision
import click_models
import interleaving

import progressbar

"""
Read queries and train click models
"""

# Read data
queries = click_models.read_click_log_file()

# Create and train random click model (RCM)
random_click_model = click_models.RandomClickModel()
random_click_model.train(queries)

# Create and train simple dependent click model (SDCM)
simple_dependent_click_model = click_models.SimpleDependentClickModel()
simple_dependent_click_model.train(queries)

"""
Interleave
"""

# p_and_e_combinations(number_of_ratings=3, length=5)

combinations = p_and_e_combinations(3, 5)
combinations_list = list(combinations)

# e_wins_team_draft_rcm = 0.0
# e_wins_prob_rcm = 0.0
# e_wins_team_draft_sdcm = 0.0
# e_wins_prob_sdcm = 0.0
results = []

bar = progressbar.ProgressBar(maxval=len(combinations_list),
                              widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

N = 100
for relevant_scores in bar(combinations_list):
    result = {}
    p = list(relevant_scores[0])
    e = list(relevant_scores[1])
    interleaved_team_draft = interleaving.team_draft_interleaving(p, 'p', e, 'e', all_unique=True)
    interleaved_prob = interleaving.prob_interleaving(p, 'p', e, 'e', all_unique=True)

    relevance_labels_team_draft = [rel for rel, _ in interleaved_team_draft]
    relevance_labels_prob = [rel for rel, _ in interleaved_team_draft]

    result['p_precision_at_5'] = precision_at_k(p, 5)
    result['e_precision_at_5'] = precision_at_k(e, 5)

    result['p_discounted_cumulative_gain_at_5'] = discounted_cumulative_gain_at_k(p, 5)
    result['e_discounted_cumulative_gain_at_5'] = discounted_cumulative_gain_at_k(e, 5)

    result['p_rank_biased_precision'] = rank_biased_precision(p)
    result['e_rank_biased_precision'] = rank_biased_precision(e)

    rcm_team_draft = 0.0
    rcm_prob = 0.0
    sdcm_team_draft = 0.0
    sdcm_prob = 0.0
    for _ in range(N):
        if interleaving.rate_interleaved(interleaved_team_draft, random_click_model.predict_clicks(
                relevance_labels_team_draft)) == 'e':
            rcm_team_draft += 1

        if interleaving.rate_interleaved(interleaved_prob,
                                         random_click_model.predict_clicks(relevance_labels_prob)) == 'e':
            rcm_prob += 1

        if interleaving.rate_interleaved(interleaved_team_draft,
                                         simple_dependent_click_model.predict_clicks(
                                             relevance_labels_team_draft)) == 'e':
            sdcm_team_draft += 1

        if interleaving.rate_interleaved(interleaved_prob, simple_dependent_click_model.predict_clicks(
                relevance_labels_prob)) == 'e':
            sdcm_prob += 1

    result['rcm-team_draft'] = rcm_team_draft / N
    result['rcm-prob'] = rcm_prob / N
    result['sdcm-team_draft'] = sdcm_team_draft / N
    result['sdcm-prob'] = sdcm_prob / N
    results.append(result)

result_data = pd.DataFrame(results)

with open('result_data.pickle', 'w') as f:
    pickle.dump(result_data, f)

print result_data.describe()

result_data[result_data['e_discounted_cumulative_gain_at_5'] < result_data['p_discounted_cumulative_gain_at_5']][
    ['rcm-prob', 'rcm-team_draft', 'sdcm-prob', 'sdcm-team_draft']].describe()
