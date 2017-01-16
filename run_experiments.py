from evaluation import p_and_e_combinations
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

e_wins_rcm = 0.0
e_wins_sdcm = 0.0
N = 0.0
for results in combinations_list:
    p = list(results[0])
    e = list(results[1])
    interleaved = interleaving.team_draft_interleaving(p, 'p', e, 'e', all_unique=True)
    if interleaving.rate_interleaved(interleaved, random_click_model.predict_clicks()) == 'e':
        e_wins_rcm += 1
    N += 1

print "--------------------------"
print "             E wins from P"
print "RCM: %21.2f" % (e_wins_rcm/N)