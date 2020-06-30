import os
import sys
import time
import argparse
import bz2
import pickle
import numpy as np
import resource
import logging
from datetime import datetime
from tqdm import trange, tqdm
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import utils
import tweetrecords
import metrics
import tf_idf

#utils.batch_tokens_to_matrix(tokens_batch, vocabulary)

parser = argparse.ArgumentParser()
parser.add_argument('--engage_mode', help='tweet: 0, retweet: 1, retweet with comment: 2, like:3', default='retweet')
parser.add_argument('--mode', help = 'train or eval: train_eval or eval', default = 'train_eval')
parser.add_argument('--model_name', help = 'change the last number for new training', default = 'RandomForest_R3')
parser.add_argument('--batch_size', help='batch size', type=int, default=500)
parser.add_argument('--data_dir', help="data directory", default='../data')
parser.add_argument('--vocab_file', help='vocabulary term document count file', default = 'document_freq_count_train.pickle')
parser.add_argument('--train_file', help = 'name of train file', default = "train_retweet_min_19.tsv")
"test_50.tsv"
parser.add_argument('--eval_file', help = 'name of validation file', default = "val_retweet_min_19.tsv")
parser.add_argument('--output_dir', help = 'output data directory', default = "../saved_models")
parser.add_argument('--log_dir', help = 'output data directory', default = "../logs")
parser.add_argument('--log_every', help = 'log data every this number of batches', default = 100)
args = parser.parse_args()

engagement_dict = {'tweet': 0, 'retweet': 1, 'retweet_comment': 2, 'like':3}

def train_eval(model, data_gen, valfile_path, TF_IDF, engagement_mode, args, train=True):

    #get the first batch of X and y data
    X,y, features_batch = next(data_gen)
    batch_size = len(X)
    
    #placeholders for storing the targets and predictions
    y_target_all = []
    y_pred_all = np.array([])
    batch_count = 1  #counts the number of batches
    samples_processed = 0
    t1 = time.time()
    while X:
        try:
            #ceate TF-IDF values of tokens
            X = np.array(TF_IDF.create_batch(X))
            y_train = np.array(y)[:,engagement_mode]  # 0  for engagement mode = reply
            
            #concat tokens with features
            X_train = np.concatenate((X, features_batch), axis = 1)

            if train:
                model.fit(X_train, y_train) # fit model
                model.n_estimators +=1 #increase number of tree by 1 for the next round
            
            y_pred = model.predict_proba(X_train) # perform prediction
            
            y_pred = y_pred[:, 1]
            
            if y_pred_all.shape[0] > 0: #if this is not first batch
                y_pred_all = np.append(y_pred_all, y_pred, axis=0) 
            else:
                y_pred_all = y_pred #for first batch
                
            y_target_all = np.append(y_target_all, y_train)

            #Calculate metrics
            t2 = time.time()
            if (batch_count%args.log_every == 0):
                print("time taken for {} samples : {}".format(args.log_every*batch_size, t2 - t1))
                rce, cross_entropy, cross_entropy_naive = metrics.relative_cross_entropy(y_target_all, y_pred_all)
                pr_auc = metrics.pr_auc(y_target_all, y_pred_all)
                logging.info("train metrics, batch #: {}, samples processed: {}, cross entropy :  {:.5f}, cross entropy naive:  {:.5f}, relative cross entropy : {:.3f},  precision-recall AUC : {:.4f}".format(batch_count, batch_count*batch_size, cross_entropy, cross_entropy_naive, rce, pr_auc))
                t1 = time.time()

                save_model(model, args, samples_processed)
                eval(model, valfile_path, TF_IDF, engagement_mode, args)

            #get the next batch of tokens, features and targets
            X,y, features_batch = next(data_gen)
            
            batch_count +=1
            samples_processed += batch_size

        except StopIteration:
            break

    #print(y_pred_all.shape, y_target_all.shape)



def eval(model, valfile_path, TF_IDF, engagement_mode, args):

    logging.info("Starting eval")
    batch_size = args.batch_size
    engagement_mode = engagement_dict[args.engage_mode]
    
    TR_val = tweetrecords.TweetRecords(valfile_path, batch_size, args.engage_mode)
    data_gen = TR_val.tokens_target_gen()

    #get the first batch of X and y data
    X,y, features_batch = next(data_gen)
        
    #placeholders for storing the targets and predictions
    y_target_all = []
    y_pred_all = np.array([])
    batch_count = 1  #counts the number of batches
    
    t1 = time.time()
    while X:
        try:
            #ceate TF-IDF values of tokens
            X = np.array(TF_IDF.create_batch(X))
            y_train = np.array(y)[:,engagement_mode]  # 0  for engagement mode = reply

            #concat tokens with features
            X_train = np.concatenate((X, features_batch), axis = 1)

            y_pred = model.predict_proba(X_train) # perform prediction
            #print(y_pred.shape)
            y_pred = y_pred[:, 1]
            
            if y_pred_all.shape[0] > 0: #if this is not first batch
                y_pred_all = np.append(y_pred_all, y_pred, axis=0) 
            else:
                y_pred_all = y_pred #for first batch
                
            y_target_all = np.append(y_target_all, y_train)

            
            #get the next batch of tokens, features and targets
            X,y, features_batch = next(data_gen)
            
            batch_count +=1

        except StopIteration:
            break
    #Calculate metrics
    
    rce, cross_entropy, cross_entropy_naive = metrics.relative_cross_entropy(y_target_all, y_pred_all)
    pr_auc = metrics.pr_auc(y_target_all, y_pred_all)
    logging.info("Val metrics, batch #: {}, samples processed: {}, cross entropy :  {:.5f}, cross entropy naive:  {:.5f}, relative cross entropy : {:.3f},  precision-recall AUC : {:.4f}".format(batch_count, batch_count*batch_size, cross_entropy, cross_entropy_naive, rce, pr_auc))
    t2 = time.time()
    logging.info("Finished  eval, time taken: {}".format(t2-t1))


def save_model(model, args, samples_processed):
        
        saved_model_dir = args.output_dir
        train_file = args.train_file.split('.')[0] #remove the .tsv from end
        model_name = args.model_name + '_' + train_file
        localtime = utils.get_current_local_time()
        filename = model_name+'_'+localtime+ '_'+str(samples_processed)+'.pickle'
        model_save_file = os.path.join(saved_model_dir, filename)
        #torch.save(self.model.state_dict(), model_dir)
        with open(model_save_file, 'wb') as handle:
            pickle.dump(model, handle)
        delete_prev_saved_model(model_name, samples_processed, args)
        

def delete_prev_saved_model(model_name, samples_processed, args):
    model_dir  = args.output_dir
    files = os.listdir(model_dir)
    
    for f_tsv in files:
        f = f_tsv.split('.')[0]
    
        if model_name in f:
            f_parsed = f.split('_')
            samples = int(f_parsed[-1])
            if samples < samples_processed:
                os.remove(os.path.join(model_dir, f_tsv))

def main():
    
    # Set the logger
    logging_dir = args.log_dir
    engage_mode = args.engage_mode
    try:
        os.makedirs(logging_dir)
    except OSError as e:
        print("loggiing directory exists")

    log_file = 'train_RandomForest_round3_' + str(engage_mode) + '.log'

    utils.set_logger(os.path.join(logging_dir, log_file))
    logging.info('model = sklearn.ensemble.RandomForestClassifier, engagement mode: Retweet')
    logging.info('Using tf_idf values of tweet tokens and other features')
    logging.info('Does not use tweet token embeddings and user ids')
    logging.info('training set: '.format(args.train_file))
    
    batch_size = args.batch_size
    data_dir = args.data_dir
    vocab_file = args.vocab_file

    logging.info('batch size = {}'.format(batch_size))

    #load the vocab file with document count frequency for each token
    vocab_file_path = os.path.join(data_dir, vocab_file)
    vocab_handle = bz2.open(vocab_file_path, 'r')
    term_document_count = pickle.load(vocab_handle)
    
    #train_file = "train_first_1000.tsv"
    train_file = args.train_file
    trainfile_path = os.path.join(data_dir, train_file)

    #train_file = "train_first_1000.tsv"
    val_file = args.eval_file
    valfile_path = os.path.join(data_dir, val_file)

    TF_IDF = tf_idf.tf_idf(term_document_count)

    print("engagement mode: {}, value: {}".format(engage_mode, engagement_dict[engage_mode]))
    TR_train = tweetrecords.TweetRecords(trainfile_path,batch_size, engage_mode)
    data_gen_train = TR_train.tokens_target_gen()

    rng = np.random.RandomState(2020)
    model = RandomForestClassifier(warm_start=True, n_estimators=1)
    model.classes_ = [0,1]

    logging.info('Starting Training')
    #train(model, data_gen, TF_IDF, engage_mode)
    if args.mode == 'train_eval':
        train_eval(model, data_gen_train, valfile_path, TF_IDF, engagement_dict[engage_mode], args, train=True)
        logging.info('Finished Training')
    elif args.mode == 'eval':
        eval(model, valfile_path, TF_IDF, engagement_mode, args)
        logging.info('Finished Evaluation') 

    TR_train.close_file()


if __name__ == "__main__":
    main()
