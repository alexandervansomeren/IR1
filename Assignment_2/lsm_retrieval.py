import pyndri
import gensim
import utils
import numpy as np
import lsm_models

def main():
    
    # Get documents
    index = pyndri.Index('index/')
    token2id, id2token, _ = index.get_dictionary()

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    print("Building word2vec")
    embedding_size = 50
    #word2vec_model = lsm_models.word2vec_model(index, embedding_size)
    model = gensim.models.Word2Vec.load('models/word2vec.model')
   
    # TODO replace block by:
    # doc_representations = word2vec_model.docs2vec(index)
    print("Building document representations")
    doc_representations = np.zeros([embedding_size, index.document_count()])        
    for d in range(index.document_base(), index.maximum_document()):
        doc = index.document(d)[1]
        docvec = np.zeros([embedding_size, len(doc)])
        for i, word_id in enumerate(doc):
            word = id2token.get(word_id,0)
            docvec[:,i] = model[word]
        doc_representations[:,d] = np.mean(docvec, axis=1)

    print("Evaluating word2vec")
    word2vec_results = {}
    for query_id, query in topics.items():
        # Build query representation as average of wordvectors
        query_tokens = query.lower().split(' ')
        wordvec = np.zeros([embedding_size, len(query_tokens)])
        for i, word in enumerate(query_tokens):
            wordvec[:,i] = model[word]
        query_representation = np.mean(wordvec, axis=1)
        # Calculate the similarity with documents
        similarity = np.zeros([index.document_count()])
        for d in range(index.document_count()):
            similarity[d] = cosine_similarity(query_representation, doc_representation[:,d])

        print(query)
        top_doc = index.document(np.argmax(similarity)+1)[1]
        line = str(' ')
        for word_id in top_doc:
            line = line + id2token[word_id] + ' '
        print(line)
        #word2vec_results[query_id] = simila
        

if __name__ == "__main__":
    main()
