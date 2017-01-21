import numpy as np

def tf_idf(tf, df):
    pass
    #query_terms = collect_query_terms(topics)

def bm25():
    pass


def construct_tf(topics, index):
    token2id, id2token, _ = index.get_dictionary()
    query_terms = collect_query_terms(topics, token2id)
    print(len(query_terms))
    print(index.document_count())
    print(type(len(query_terms)))
    print(type(index.document_count()))
    tf = np.zeros(len(query_terms), index.document_count())
    for doc_id in range(index.document_base(), index.maximum_document()):
        for term_id, term in enumerate(query_terms):
            if term in index.document(doc_id)[1]:
                tf[doc_id-1,term_id] += 1


def construct_df(tf):
    pass

def collect_query_terms(topics, token2id):
    query_terms = set()
    for query_id, query_tokens in topics.items():
        query_id_tokens = [token2id.get(query_token,0) for query_token in query_tokens]
        query_id_tokens = [word_id for word_id in query_id_tokens if word_id > 0]
        query_terms |= set(query_id_tokens)
    return query_terms

