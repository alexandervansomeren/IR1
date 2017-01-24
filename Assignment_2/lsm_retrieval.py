import pyndri
import gensim
import utils
import lsm_models

def main():
    
    # Get documents
    index = pyndri.Index('index/')

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    word2vec_models = lsm_models.word2vec_model(index)



if __name__ == "__main__":
    main()
