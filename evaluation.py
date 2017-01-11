import itertools as itt

import itertools
import numpy as np
import matplotlib.pyplot as plt
import random

"""
---------------------------------------------------------------------------------------------
Step 1
---------------------------------------------------------------------------------------------
"""


def p_and_e_combinations(number_of_ratings=3, length=5):
    p = itertools.product(range(number_of_ratings), repeat=length)  # where 0 = N, 1 = R and 3 = HR
    e = itertools.product(range(number_of_ratings), repeat=length)  # where 0 = N, 1 = R and 3 = HR
    return itertools.product(p, e)


combinations = p_and_e_combinations()

combinations_list = list(combinations)

# combinations_sample = random.sample(list(combinations), 1000)

# print list(combinations)


"""
---------------------------------------------------------------------------------------------
Step 2
---------------------------------------------------------------------------------------------
"""

"""
Binary evaluation measures
"""


def precision_at_k(ranking, k):
    return np.divide((np.array(ranking) > 0)[0:k].sum(), float(k))


def recall_at_k(ranking, k):
    boolean_ranking = (np.array(ranking) > 0)
    number_of_relevant = boolean_ranking.sum()
    return np.divide(boolean_ranking[0:k].sum(), float(number_of_relevant))


def average_precision(ranking):
    boolean_ranking = (np.array(ranking) > 0)
    number_of_relevant = boolean_ranking.sum()
    av_precision = 0
    for rank, relevant in enumerate(boolean_ranking):
        # print rank
        # print relevant
        if relevant:
            av_precision += np.divide(boolean_ranking[0:rank].sum(), float(rank + 1))  # precision at k
    return np.divide(av_precision, number_of_relevant)


# print combinations_list[100][1]
# print average_precision(combinations_list[100][1])

"""
Multi-graded evaluation measures
"""


def discounted_cumulative_gain_at_k(ranking, k):
    boolean_ranking = np.array((np.array(ranking) > 0), dtype=int)
    dcg = 0
    for rank, relevant in enumerate(boolean_ranking[0:k]):
        dcg += float(2 ** relevant - 1) / np.log2(rank + 2)
    return dcg


def rank_biased_precision(ranking, p=0.8):
    measure = 0
    for rank, rel in enumerate(ranking):
        measure += rel * p ** rank
    return measure * (1 - p)


"""
Expected Reciprocal Rank (ERR)
"""


def prob(rank, gmax):
    return float(2 ** rank - 1) / 2 ** gmax


def expected_reciprocal_rank(ranking):
    measure = 0
    n = len(ranking)
    gmax = 4
    for r in range(1, n):
        stopProb = 1.0
        for i in range(1, r):
            stopProb *= (1 - prob(ranking[i], gmax)) * prob(ranking[r], gmax)
        measure += 1.0 / r * stopProb
    return measure


"""
---------------------------------------------------------------------------------------------
Step 3
---------------------------------------------------------------------------------------------
"""

precision_at_4_won_by_e = []
for rankings in p_and_e_combinations():
    ranking_p = rankings[0]
    ranking_e = rankings[1]

    if precision_at_k(ranking_e, k=4) > precision_at_k(ranking_p, k=4):
        precision_at_4_won_by_e.append(rankings)


def show_results(result_list, number=10):
    print "       P       ||        E      "
    print "--------------------------------"
    # print "(0, 0, 0, 0, 0)||(0, 0, 0, 1, 0)"
    result_list = random.sample(result_list, number)
    for i, result in enumerate(result_list):
        print str(result[0]) + '||' + str(result[1])

print "\nPrecision at rank 4 won by e: \n"
print show_results(precision_at_4_won_by_e, 20)
