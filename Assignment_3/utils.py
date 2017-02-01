import numpy as np


class normalized_discounted_cumulative_gain_at_k:
    idcg = 1

    def __init__(self, optimal_ranking, k):
        self.optimal_ranking = optimal_ranking
        self.idcg = self.compute(optimal_ranking, k)
        self.k = k

    def compute(self, ranking, k=0, normalize=True):
        if k == 0:
            k = self.k
        dcg = 0.0
        for rank, relevance in enumerate(ranking[0:k]):
            dcg += float(2 ** relevance - 1) / np.log2(rank + 2)
        if normalize:
            dcg /= self.idcg
        if dcg > 1:
            dcg = 1.0
        return dcg