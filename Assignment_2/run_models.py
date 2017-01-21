import pyndri
import utils
import models

def main():

    # Get documents
    index = pyndri.Index('index/')

    # Get queries
    with open('./ap_88_89/topics_title', 'r') as f_topics:
        topics = utils.parse_topics(f_topics)

    # Construct term frequency and document frequency
    tf = models.construct_tf(topics, index)
    #df = models.construct_df(tf)

    # Run models
    #tf_idf = models.tf_idf(tf, df)
    #bm25 = models.bm25(tf, df)

if __name__ == "__main__":
    main()




