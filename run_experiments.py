import pandas as pd
import cPickle as pickle

from evaluation import p_and_e_combinations
from evaluation import precision_at_k
from evaluation import discounted_cumulative_gain_at_k
from evaluation import rank_biased_precision
import click_models
import interleaving

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

for relevant_scores in combinations_list[0:len(combinations_list)]:
    result = {}
    p = list(relevant_scores[0])
    e = list(relevant_scores[1])
    interleaved_team_draft = interleaving.team_draft_interleaving(p, 'p', e, 'e', all_unique=True)
    interleaved_prob = interleaving.prob_interleaving(p, 'p', e, 'e', all_unique=True)

    result['rcm-team_draft'] = interleaving.rate_interleaved(interleaved_team_draft, random_click_model.predict_clicks())
    result['rcm-prob'] = interleaving.rate_interleaved(interleaved_prob, random_click_model.predict_clicks())

    result['p_precision_at_5'] = precision_at_k(p, 5)
    result['e_precision_at_5'] = precision_at_k(e, 5)



    result['p_discounted_cumulative_gain_at_5'] = discounted_cumulative_gain_at_k(p, 5)
    result['e_discounted_cumulative_gain_at_5'] = discounted_cumulative_gain_at_k(e, 5)

    result['p_rank_biased_precision'] = rank_biased_precision(p)
    result['e_rank_biased_precision'] = rank_biased_precision(e)



    results.append(result)


result_data = pd.DataFrame(results)

with open('result_data.pickle', 'w') as f:
    pickle.dump(results, f)


print result_data.describe()
# print "------------------------------------"
# print "                       E wins from P"
# print "RCM-team_draft : %19.4f" % (e_wins_team_draft_rcm/N)
# print "RCM-prob       : %19.4f" % (e_wins_prob_rcm/N)