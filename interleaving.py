import random
import operator
from numpy.random import choice
import numpy as np

p = ['a', 'b', 'c', 'd']
e = ['a', 'c', 'b', 'e']


def team_draft_interleaving(serp1, serp1_name, serp2, serp2_name, all_unique=False):
    interleaved = []
    serp1 = list(serp1)
    serp2 = list(serp2)

    while serp1 != [] and serp2 != []:
        if random.randint(0, 1) == 0:
            result = serp1.pop(0)
            interleaved.append((result, serp1_name))
            if result in serp2 and not all_unique:
                serp2.remove(result)
        else:
            result = serp2.pop(0)
            interleaved.append((result, serp2_name))
            if result in serp1 and not all_unique:
                serp1.remove(result)

    # Extend the rest if one is empty
    interleaved.extend([(s, serp1_name) for s in serp1])
    interleaved.extend([(s, serp2_name) for s in serp2])
    return interleaved


# print team_draft_interleaving(p, 'p', e, 'e')


def rate_interleaved(interleaved, click_pattern):
    names = list(set([res[1] for res in interleaved]))
    results = {name: 0 for name in names}

    for rank, click in enumerate(click_pattern):
        if click:
            results[interleaved[rank][1]] += 1

    if results[names[0]] == results[names[1]]:
        return "tie"
    if results[names[0]] > results[names[1]]:
        return names[0]
    else:
        return names[1]


# interleaved = team_draft_interleaving(p, 'p', e, 'e')
# print interleaved
# click_pattern = [0, 0, 1, 1, 0]
# #
# p = (1, 0, 1, 2, 0)
# e = (0, 0, 1, 1, 0)

# interleaved2 = team_draft_interleaving(p, 'p', e, 'e', all_unique=True)
# print interleaved2


"""
Probabilistic interleaving
"""


def prob_interleaving(serp1, serp1_name, serp2, serp2_name, all_unique=False):
    tau = 3.0

    serp1 = list(serp1)
    serp2 = list(serp2)

    normalization_factor = 0.0
    for rank, _ in enumerate(serp1):
        normalization_factor += 1 / (rank + 1) ** tau
    p = []
    for rank, _ in enumerate(serp1):
        p.append((1.0 / (rank + 1) ** tau) / normalization_factor)

    p_serp1 = np.array(p)
    p_serp2 = np.array(p[:])
    interleaved = []

    while serp1 != [] and serp2 != []:
        if random.randint(0, 1) == 0:
            p_serp1 = p_serp1 / sum(p_serp1)  # normalize new distribution

            draw = choice(serp1, 1, p=p_serp1)
            interleaved.append((draw[0], serp1_name))

            p_serp1 = np.delete(p_serp1, serp1.index(draw))
            serp1.remove(draw)

            if draw in serp2 and not all_unique:
                p_serp2 = np.delete(p_serp2, serp2.index(draw))
                serp2.remove(draw)
        else:
            p_serp2 = p_serp2 / sum(p_serp2)

            draw = choice(serp2, 1, p=p_serp2)
            interleaved.append((draw[0], serp2_name))

            p_serp2 = np.delete(p_serp2, serp2.index(draw))
            serp2.remove(draw)
            if draw in serp1 and not all_unique:
                p_serp1 = np.delete(p_serp1, serp1.index(draw))
                serp1.remove(draw)

    while serp1:
        if len(serp1) == 1:
            interleaved.append((serp1[0], serp1_name))
            break
        p_serp1 = p_serp1 / sum(p_serp1)
        draw = choice(serp1, 1, p=p_serp1)
        interleaved.append((draw[0], serp1_name))
        p_serp1 = np.delete(p_serp1, serp1.index(draw))
        p_serp1 = p_serp1 / sum(p_serp1)  # normalize new distribution
        serp1.remove(draw)

    while serp2:
        if len(serp2) == 1:
            interleaved.append((serp2[0], serp2_name))
            break
        p_serp2 = p_serp2 / sum(p_serp2)
        draw = choice(serp2, 1, p=p_serp2)
        interleaved.append((draw[0], serp2_name))
        p_serp2 = np.delete(p_serp2, serp2.index(draw))
        serp2.remove(draw)

    return interleaved


# interleaved2 = prob_interleaving(p, 'p', e, 'e', all_unique=True)
# print interleaved2
