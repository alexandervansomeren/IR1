import numpy as np


def tf_idf(tf):
    if 0 in tf:
        tf += 0.001
    df = tf.sum(axis=1)
    n_docs = df.shape(2)
    return np.log(1 + tf).T * np.log(n_docs / df.T)  # tf-idf


def bm25():
    pass


def cosine_similarity(tf_idf, query):
    return np.einsum('ij,i->j', tf_idf, query) / np.linalg.norm(tf_idf, axis=0) * np.linalg.norm(query)


def construct_tf(topics, index, max_query_terms=0, max_documents=0):
    token2id, id2token, _ = index.get_dictionary()
    query_terms = collect_query_terms(topics, token2id)
    if max_query_terms > 0:
        query_terms = query_terms[0:max_query_terms]
    n_docs = index.document_count()
    if max_documents > 0:
        n_docs = max_documents
    tf = np.zeros([len(query_terms), n_docs])
    for doc_id in range(n_docs):  # doc_id is shifted to the right in tf matrix (1 -> 0)
        for term_id, term in enumerate(query_terms):
            if term in index.document(doc_id + 1)[1]:
                tf[term_id, doc_id] = index.document(doc_id + 1)[1].count(term)
    return tf


def collect_query_terms(topics, token2id):
    query_terms = set()
    for query_id, query_tokens in topics.items():
        query_tokens = query_tokens.lower().split(' ')
        query_id_tokens = [token2id.get(query_token, 0) for query_token in query_tokens]
        query_id_tokens = [word_id for word_id in query_id_tokens if word_id > 0]
        query_terms |= set(query_id_tokens)
    return query_terms
