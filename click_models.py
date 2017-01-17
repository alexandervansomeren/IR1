import random

"""
---------------------------------------------------------------------------------------------
Read Data
---------------------------------------------------------------------------------------------
"""


def read_click_log_file(filename="YandexRelPredChallenge.txt"):
    data = open(filename, 'r')

    queries = []
    for line in data:
        entry_array = line.strip().split("\t")

        if entry_array[2] == "Q":
            doc_urls = entry_array[5::]
            click_pattern = []
            query = {"query_id": entry_array[3],
                     "doc_urls": doc_urls,
                     "click_pattern": click_pattern}
            queries.append(query)

        elif entry_array[2] == "C":
            clicked_doc = entry_array[3]
            if clicked_doc in doc_urls:
                click_pattern.append(clicked_doc)
                # [doc_urls.index(clicked_doc)] = 1

    return queries


"""
---------------------------------------------------------------------------------------------
Random Click Model
---------------------------------------------------------------------------------------------
"""


class RandomClickModel():
    def __init__(self):
        self.rho = 0.0

    def train(self, queries):
        amount_of_clicks = 0
        self.serp_length = len(queries[0]['doc_urls'])
        for query in queries:
            amount_of_clicks += len(query["click_pattern"])
        amount_of_shown_docs = len(queries) * 10
        self.rho = float(amount_of_clicks) / amount_of_shown_docs

    def predict_clicks(self, relevance_labels_list):
        clicks = []
        for _ in range(len(relevance_labels_list)):
            clicks.append(int(random.random() < self.rho))
        return clicks


"""
---------------------------------------------------------------------------------------------
Simple Dependent Click Model
---------------------------------------------------------------------------------------------
"""


class SimpleDependentClickModel():
    def __init__(self):
        self.lambdas = []

    def train(self, queries):
        lambdas = []
        for r in range(10):
            rank_clicked_counter = 0
            rank_not_last_clicked_counter = 0
            for query in queries:
                # Count if doc at rank r is clicked
                doc_at_rank = query["doc_urls"][r]
                if doc_at_rank in query["click_pattern"]:
                    rank_clicked_counter += 1
                    # Count if last clicked rank is not equal to r
                    if len(query["click_pattern"]) - 1 != r:
                        rank_not_last_clicked_counter += 1
            l = rank_not_last_clicked_counter / float(rank_clicked_counter)
            lambdas.append(l)
        self.lambdas = lambdas

    def predict_clicks(self, relevance_labels_list, max_relevance=2):
        clicks = []
        epsilon = 1
        alpha = [relevance / float(max_relevance+1) for relevance in relevance_labels_list]
        for r in range(len(relevance_labels_list)):
            prob = alpha[r] * epsilon
            clicks.append(int(random.random() < prob))
            if sum(clicks) > 0:
                epsilon = (clicks[r] * self.lambdas[r] + (1 - clicks[r]) *
                           (1 - alpha[r]) * epsilon / float(1 - alpha[r] * epsilon))
            else:
                epsilon = epsilon * (alpha[r] * self.lambdas[r] + (1 - alpha[r]))
        return clicks


"""
    def attractiveness(self, relevance_labels_list):
        alpha = []
        for relevance in relevance_labels_list:
            if   relevance == 0: alpha.append(0)
            elif relevance == 1: alpha.append(0.5)
            elif relevance == 2: alpha.append(1)
        return alpha
"""

# ranking = [0, 1, 1, 2, 0]
# queries = read_click_log_file()
# print(queries[4])
# rho = rcm_train_params(queries)
# probabilities = rcm_probabilities(ranking, rho)
# clicks = rcm_predict_clicks(probabilities)
