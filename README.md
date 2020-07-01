# insight_tweet_prediction
Prediction of probability of Retweet engagement modes to tweets displayed on Home Timeline

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install transformers
```

### Directory Structure 

|-- /`code` : contains all python files

|-- `/saved_features:` Tweet features and text embeddings for each record

|-- `/data:` Train and validation files with tweet records

|-- `/saved_models:` pre-trained models

|-- `/output:` Where outputs will be saved

|-- `/logs:` folder to save log files



#### RandomForest model using token TF-IDF and tweet features

<img src="/images/RF.png" width = "350" height="300">

*  **`python train_RandomForest.py --mode=eval:`** train file to run RandomForest model

*  **`tf_idf.py:`** class to generate TF-IDF of tweet tokens

#### Multilayer Neural Network Model

<img src="/images/MLP.png" width = "350" height="300">

* **`MLP_Baseline.py`** Multilayerd NN model

#### Neural Collaborative Filtering

<img src="/images/NCF.png" width = "400" height="300">

* **`NCF_MLP.py`** model implementing Neural Collaborative Filtering 

* **`engine.py`** engine that trains either Multilayer NN or Neural Collaborative Filtering model

* **`python train_tweet_NN.py  --mode=eval`** file that loads the parameters and executes the training and eval engine

### Files utilized by all 3 models

*  **`tweetrecords.py :`**  class that handles all operations related to tweets before they are ready for input to model

*  **`utils.py:`** contains helper functions

*  **`metrics.py:`**  methods for calculating the metrics

## data directory contains the necessary files to run the training models

*  **test50:** file containing sample of 50 records
