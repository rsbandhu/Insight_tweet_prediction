# insight_tweet_prediction
Prediction of probability of Retweet engagement modes to tweets displayed on Home Timeline

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install transformers
```

### Directory Structure 

|-- /`code` : contains all python files

|-- `/saved_models:` pre-trained models

|-- `/data:` Train and validation files with tweet records

|-- `/output:` Where outputs will be saved

|-- `/logs:` folder to save log files

### contains the necessary files to run the training models

*  **`tweetrecords.py :`**  class that handles all operations related to tweets before they are ready for input to model

*  **`utils.py:`** contains helper functions

*  **`metrics.py:`**  methods for calculating the metrics

*  **`tf_idf.py:`** class to generate TF-IDF of tweet tokens

#### RandomForest model using token TF-IDF and tweet features

![RandomForest Model](/images/RF.png)

*  **`python train_RandomForest.py --mode=eval:`** train file to run RandomForest model

#### Neural Network Models

* **`MLP_Baseline.py`** Multilayerd neural netowrk model

* **`NCF_MLP.py`** model implementing Neural Collaborative 

* **`engine.py`** engine that trains either MLP_Baseline or NCF_MLP

* **`train_tweet_NN.py`** file that loads the parameters and executes the training and eval engine


## data directory contains the necessary files to run the training models

*  **test50:** file containing sample of 50 records
