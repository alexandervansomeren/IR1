import os
import argparse
import pyndri
import gensim
import utils
import numpy as np
import lsm_models

FLAGS = None

def run_w2v(index, doc_names, topics, embedding_size, max_documents):
    
    print("Building / loading word2vec")
    wv2_model_filename = 'models/word2vec.model'    
    if os.path.isfile(wv2_model_filename):
        w2v = lsm_models.Word2Vec(filename=wv2_model_filename,
                                  embedding_size=embedding_size,
                                  max_documents=max_documents)
    else:
        w2v = lsm_models.Word2Vec(embedding_size=embedding_size,
                                  max_documents=max_documents) 
        w2v.train(index)

    print('Size Word2Vec model')
    print(len(w2v.model.wv.vocab))

    print("Building document representations")
    docs_representation_filename = 'tmp/doc2vecs.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = w2v.docs2vec(index)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation)

    print("Scoring documents")
    w2v_results = {}
    for query_id, query in topics.items():
        # Get query word2vec representation
        query_representation = w2v.query2vec(query)
        # Calculate the similarity with documents
        w2v_score = utils.cosine_similarity(query_representation, docs_representation)
        w2v_results[query_id] = list(zip(w2v_score, doc_names))

        #print(query)
        #top_doc = index.document(np.argmax(w2v_score)+1)[1]
        #line = str(' ')
        #for word_id in top_doc:
        #    line = line + str(id2token.get(word_id,0)) + ' '
        #print(line)

    # Save results to file
    utils.write_run(model_name='w2v', data=w2v_results, 
                    out_f='results/ranking_w2v.txt', max_objects_per_query=1000)


def run_lsi(index, doc_names, topics, num_topics):
    
    print("Building / loading LSI")
    lsi_model_filename = 'models/lsi.model'    
    if os.path.isfile(lsi_model_filename):
        lsi = lsm_models.LSI(filename=lsi_model_filename,
                             num_topics=num_topics)
    else:
        lsi = lsm_models.LSI(num_topics=num_topics) 
        lsi.train(index)

    #print('Size Word2Vec model')
    #print(len(w2v.model.wv.vocab))

    print("Scoring documents")
    lsi_results = {}
    #for query_id, query in topics.items():
        # Get query word2vec representation
        #query_representation = w2v.query2vec(query)
        # Calculate the similarity with documents
        #w2v_score = utils.cosine_similarity(query_representation, docs_representation)
        #w2v_results[query_id] = list(zip(w2v_score, doc_names))


    # Save results to file
    #utils.write_run(model_name='w2v', data=w2v_results, 
    #                out_f='results/ranking_w2v.txt', max_objects_per_query=1000)



def initialize_folders():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('results'):
        os.makedirs('results')


def main():

    # Get documents
    index = pyndri.Index('index/')
    token2id, id2token, _ = index.get_dictionary()
    doc_names = utils.get_document_names(index)

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    # Create directories if they do not exist
    initialize_folders()

    # Run LSM for command line argument method
    if FLAGS.method == 'word2vec':
        run_w2v(index, doc_names, topics, 300, index.document_count())
    elif FLAGS.method == 'lsi':
        run_lsi(index, doc_names, topics, 20)


if __name__ == "__main__":

    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--method', type = str, default = 'lsi',
                        help='Latent semanctic model [word2vec, lsi, lda, doc2vec].')

    FLAGS, unparsed = parser.parse_known_args()

    main()
