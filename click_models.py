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
      query = { "query_id": entry_array[3],
                "doc_urls": doc_urls,
                "click_pattern": click_pattern }
      queries.append(query)

    elif entry_array[2] == "C":
      clicked_doc = entry_array[3]
      if clicked_doc in doc_urls:
        click_pattern.append(clicked_doc)
        #[doc_urls.index(clicked_doc)] = 1

  return queries


"""
---------------------------------------------------------------------------------------------
Random Click Model
---------------------------------------------------------------------------------------------
"""

def rcm_train_params(queries):
  amount_of_clicks = 0
  for query in queries:
    amount_of_clicks += len(query["click_pattern"])
  amount_of_shown_docs = len(queries)*10
  return float(amount_of_clicks) / amount_of_shown_docs

def rcm_probabilities(ranking, rho):
  return len(ranking) * [rho]

def rcm_predict_clicks(probabilities):
  clicks = []
  for p in probabilities:
    clicks.append(random.random() < p)
  return clicks   

"""
---------------------------------------------------------------------------------------------
Simple Dependent Click Model
---------------------------------------------------------------------------------------------
"""

def sdcm_train_params(queries):
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
        if len(query["click_pattern"])-1 != r:
          rank_not_last_clicked_counter += 1
    l = rank_not_last_clicked_counter / float(rank_clicked_counter)
    lambdas.append(l)
  return lambdas

def sdcm_probabilities(ranking, lambdas):


ranking = [0,1,1,2,0]
queries = read_click_log_file()
print(queries[4])
rho = rcm_train_params(queries)
probabilities = rcm_probabilities(ranking,rho)
clicks = rcm_predict_clicks(probabilities)

