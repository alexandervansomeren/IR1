import collections
import io
import logging
import sys
import numpy as np

def parse_topics(f,
                 max_topics=sys.maxsize, delimiter=';'):
    assert max_topics >= 0 or max_topics is None

    topics = collections.OrderedDict()

    for line in f:
        assert(isinstance(line, str))

        line = line.strip()

        if not line:
            continue

        topic_id, terms = line.split(delimiter, 1)

        if topic_id in topics and (topics[topic_id] != terms):
                logging.error('Duplicate topic "%s" (%s vs. %s).',
                              topic_id,
                              topics[topic_id],
                              terms)

        topics[topic_id] = terms

        if max_topics > 0 and len(topics) >= max_topics:
            break

    return topics

def write_run(model_name, data, out_f,
              max_objects_per_query=sys.maxsize,
              skip_sorting=False):
    """
    Write a run to an output file.
    Parameters:
        - model_name: identifier of run.
        - data: dictionary mapping topic_id to object_assesments;
            object_assesments is an iterable (list or tuple) of
            (relevance, object_id) pairs.
            The object_assesments iterable is sorted by decreasing order.
        - out_f: output file stream.
        - max_objects_per_query: cut-off for number of objects per query.
    """
    f = open(out_f, "w")

    for subject_id, object_assesments in data.items():
        if not object_assesments:
            logging.warning('Received empty ranking for %s; ignoring.',
                            subject_id)

            continue

        # Probe types, to make sure everything goes alright.
        # assert isinstance(object_assesments[0][0], float) or \
        #     isinstance(object_assesments[0][0], np.float32)
        assert isinstance(object_assesments[0][1], str) or \
            isinstance(object_assesments[0][1], bytes)

        if not skip_sorting:
            object_assesments = sorted(object_assesments, reverse=True)

        if max_objects_per_query < sys.maxsize:
            object_assesments = object_assesments[:max_objects_per_query]

        if isinstance(subject_id, bytes):
            subject_id = subject_id.decode('utf8')

        for rank, (relevance, object_id) in enumerate(object_assesments):
            if isinstance(object_id, bytes):
                object_id = object_id.decode('utf8')

            f.write(
                '{subject} Q0 {object} {rank} {relevance} '
                '{model_name}\n'.format(
                    subject=subject_id,
                    object=object_id,
                    rank=rank + 1,
                    relevance=relevance,
                    model_name=model_name))


def cosine_similarity(query, docs):
    """
    :param tf_idf: tf_idf matrix where rows are terms and columns are documents
    :param query: query representation vector 
    :param docs: docs representation matrix where rows are terms or embeddings
                 and columns are documents
    :return: numpy array of cosine similarity per document
    """
    return np.einsum('ij,i->j',docs,query) / np.linalg.norm(docs,axis=0) * np.linalg.norm(query)


def get_document_names(index):
    doc_names = []
    for i in range(index.document_base(), index.maximum_document()):
        doc_names.append(index.document(i)[0])
    return doc_names

def get_top_1000_tf_idf(topics):
    # Load tf-idf model
    with open('tfidf.npy', 'rb') as f:
        tf_idf = np.load(f)
    # Load word to tf-idf index dict
    with open('term2index.json', 'r') as f:
        term2index = json.load(f)
    # Get top 1000 documents tf-idf ranking
    for query_id, query in topics.items():
        query_indices = models.query2indices(query, term2index)
        tf_idf_score = models.tf_idf_score(tf_idf, query_indices)
        tf_idf_ranked_doc_indices = np.argsort(-tf_idf_score)
        best_1000_doc_indices = tf_idf_ranked_doc_indices[0:1000]


