import random
import operator

p = ['a', 'b', 'c', 'd']
e = ['a', 'c', 'b', 'e']


def team_draft_interleaving(serp1, serp1_name, serp2, serp2_name):
    interleaved = []

    while serp1 != [] and serp2 != []:
        if random.randint(0, 1) == 0 and serp1 != []:
            result = serp1.pop(0)
            interleaved.append((result, serp1_name))
            if result in serp2:
                serp2.remove(result)
        elif serp2:
            result = serp2.pop(0)
            interleaved.append((result, serp2_name))
            if result in serp1:
                serp1.remove(result)

    # Extend the rest if one is empty
    interleaved.extend([(s, serp1_name) for s in serp1])
    interleaved.extend([(s, serp2_name) for s in serp2])
    return interleaved


# print team_draft_interleaving(p, 'p', e, 'e')


def rate_team_draft_interleaved(interleaved, click_pattern):
    names = list(set([res[1] for res in interleaved]))
    results = {name: 0 for name in names}

    for rank, click in enumerate(click_pattern):
        if click:
            print "click!"
            results[interleaved[rank][1]] += 1

    if results[names[0]] == results[names[1]]:
        return "tie"
    if results[names[0]] > results[names[1]]:
        return names[0]
    else:
        return names[1]


interleaved = team_draft_interleaving(p, 'p', e, 'e')
print interleaved
click_pattern = [0, 0, 1, 1, 0]

print rate_team_draft_interleaved
