import os
import pyndri
import gensim
import utils
import numpy as np
import lsm_models

def main():
    
    # Get documents
    index = pyndri.Index('index/')
    token2id, id2token, _ = index.get_dictionary()
    doc_names = utils.get_document_names(index)

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    initialize_folders()

    print("Building / loading word2vec")
    embedding_size = 100
    max_documents = 2000
    wv2_model_filename = 'models/word2vec.model'    
    if os.path.isfile(wv2_model_filename):
        w2v = lsm_models.Word2Vec(filename=wv2_model_filename,
                                  embedding_size=embedding_size,
                                  max_documents=max_documents)
    else:
        w2v = lsm_models.Word2Vec() 
        w2v.train(index)
    #w2v_model = lsm_models.word2vec_model(index, embedding_size)
    #w2v_model = gensim.models.Word2Vec.load('models/word2vec.model')
    print('Size Word2Vec model')
    print(len(w2v.model.vocab))
   
    print("Building document representations")
    docs_representation_filename = 'tmp/doc2vecs.npy'
    if os.path.isfile(docs_representation_filename):
        with open(docs_representation_filename, 'rb') as f:
            docs_representation = np.load(f)
    else:
        docs_representation = w2v.docs2vec(index, id2token)
        with open(docs_representation_filename, 'wb') as f:
            np.save(f, docs_representation)
#     doc_representations = np.zeros([embedding_size, index.document_count()])        
#     for d in range(index.document_base(), index.maximum_document()):
#         doc = index.document(d)[1]
#         doc_words = list((id2token.get(word_id) for word_id in doc if(word_id != 0 and id2token.get(word_id) in model.vocab)))
#         docvec = np.zeros([embedding_size, len(doc_words)])
#         for i, word in enumerate(doc_words):
#             docvec[:,i] = model[word]
#         doc_representations[:,d] = np.mean(docvec, axis=1)

    print("Scoring documents")
    w2v_results = {}
    for query_id, query in topics.items():
        # Get query word2vec representation
        query_representation = w2v.query2vec(query, id2token)
        # Calculate the similarity with documents
        w2v_score = utils.cosine_similarity(query_representation, docs_representation)
        w2v_results[query_id] = list(zip(w2v_score, doc_names))

        print(query)
        top_doc = index.document(np.argmax(w2v_score)+1)[1]
        line = str(' ')
        for word_id in top_doc:
            line = line + str(id2token.get(word_id,0)) + ' '
        print(line)

    utils.write_run(model_name='w2v', data=w2v_results, 
                    out_f='results/ranking_w2v.txt', max_objects_per_query=1000)

        

def initialize_folders():
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    if not os.path.exists('results'):
        os.makedirs('results')

if __name__ == "__main__":
    main()
