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
      click_pattern = 10*[0]
      query = { "query_id": entry_array[3],
                "doc_urls": doc_urls,
                "click_pattern": click_pattern }
      queries.append(query)

    elif entry_array[2] == "C":
      clicked_doc = entry_array[3]
      if clicked_doc in doc_urls:
        click_pattern[doc_urls.index(clicked_doc)] = 1

  return queries


"""
---------------------------------------------------------------------------------------------
Random Click Model
---------------------------------------------------------------------------------------------
"""

def rcm_train_params(queries):
  amount_of_clicks = 0
  for query in queries:
    amount_of_clicks += sum(query["click_pattern"])
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
TODO OTHER MODEL
---------------------------------------------------------------------------------------------
"""

ranking = [0,1,1,2,0]
queries = read_click_log_file()
rho = rcm_train_params(queries)
probabilities = rcm_probabilities(ranking,rho)
clicks = rcm_predict_clicks(probabilities)

