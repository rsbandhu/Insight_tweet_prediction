# insight_tweet_prediction
Prediction of probability of different engagement modes to tweets displayed on Home Timeline


## code directory contains the necessary files to run the training models

*  **tweetrecords.py :**  contains the class that handles all operations related to tweets before they are ready for input to model

*  **utils.py:** contains helper functions

*  **metrics.py:**  methods for calculating the metrics

*  **tf_idf.py:** class to generate TF-IDF of tweet tokens

*  **tweet_RandomForest.py:** train file to run RandomForest model

*  **tweet_NaiveBayes.py:** train file to run Naive-Bayes using tweet tokens


## data directory contains the necessary files to run the training models

*  **test50:** file containing sample of 50 records
