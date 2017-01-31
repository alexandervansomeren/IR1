#### Code by Alexander van Someren & Ilse van der Linden

##### Task 1 
Run [run_models.py]: creates tf and df matrices 
Run [evaluate_models.py]: rank all models in task 1
Run [plm.py]: rank plm in task 1

##### Task 2
Run [lsm_retrieval.py --method ['word2vec', 'lsi', 'lda', 'doc2vec']]: 
builds specific model for range of hyperparameters [50,100,150,200]
re-ranks top 1000 tf-idf ranked documents with the model

##### Results
Run [run_parse_eval.py]: evaluate all rankings in results folders
