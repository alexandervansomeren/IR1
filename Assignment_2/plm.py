import numpy as np


def construct_tf(topics, index, max_query_terms=0, max_documents=0):
    token2id, id2token, _ = index.get_dictionary()
    query_term_ids = collect_query_terms(topics, token2id)

    n_docs = index.document_count()
    # Take small sample for debugging purposes
    if max_query_terms > 0:
        query_term_ids = list(query_term_ids)[0:max_query_terms]
    if max_documents > 0:
        n_docs = max_documents

    # Count term frequencies per query term for each document
    tf = np.zeros([len(query_term_ids), n_docs])
    for doc_id in range(n_docs):  # doc_id is shifted to the right in tf matrix (1 -> 0)
        for term_id, term in enumerate(query_term_ids):
            if term in index.document(doc_id + 1)[1]:
                tf[term_id, doc_id] = index.document(doc_id + 1)[1].count(term)

    # Create dictionary to retrieve index of term in tf matrix
    term2index = {id2token[term_id]: index for index, term_id in enumerate(query_term_ids)}
    return tf, term2index