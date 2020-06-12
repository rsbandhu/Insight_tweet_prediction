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
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import utils
import tweetrecords
import metrics

#utils.batch_tokens_to_matrix(tokens_batch, vocabulary)

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch size', type=int, default=1000)
parser.add_argument('--engage_mode', help='tweet: 0, retweet: 1, retweet with comment: 2, like:3', type=int, default=3)
parser.add_argument('--data_dir', help="data directory", default='../data')
parser.add_argument('--vocab_file', help='vocabulary file', default = 'vocab_training.pickle')
parser.add_argument('--train_file', help = 'name of train file', default = "training_train.tsv")
parser.add_argument('--eval_file', help = 'name of validation file', default = "training_validation.tsv")
parser.add_argument('--model_save_file', help = 'name of file to save trained model', default = "../output/train_Multinomial.pickle")
parser.add_argument('--output_dir', help = 'output data directory', default = "../output")
parser.add_argument('--log_dir', help = 'output data directory', default = "../logs")
parser.add_argument('--log_every', help = 'log data every this number of batches', default = 5)

args = parser.parse_args()

def train_eval(model, data_gen, vocab, engagement_mode, train=True):

    vocab_size  = len(vocab)
    #get the first batch of X and y data
    X,y = next(data_gen)
    batch_size = len(X)
    #Convert X to matcrix
    #print("before first train batch, vocab size: ",len(vocab))
    X_train = utils.batch_tokens_to_matrix(X,vocab, vocab_size)
    y_train = np.array(y)[:,engagement_mode]  # 0  for engagement mode = reply

    print(X_train.shape, y_train.shape)
    #placeholders for storing the targets and predictions
    y_target_all = []
    y_pred_all = np.array([])
    batch_count = 1  #counts the number of batches
    
    t1 = time.time()
    while X:
        try:
            print("train batch beginning, vocab size: ",len(vocab))
            if train:
                #print(X_train.shape)
                model.partial_fit(X_train, y_train, [0,1], 1.)
            
            y_pred = model.predict_proba(X_train)[:, 1]

            if y_pred_all.shape[0] > 0: #if this is not first batch
                y_pred_all = np.append(y_pred_all, y_pred, axis=0) 
            else:
                y_pred_all = y_pred #for first batch
                
            y_target_all = np.append(y_target_all, y_train)

            #Calculate metrics
            if (batch_count%args.log_every == 0):
                t2 = time.time()
                print("time taken for {} samples : {}".format(args.log_every *batch_size, t2 - t1))
                rce, cross_entropy, cross_entropy_naive = metrics.relative_cross_entropy(y_target_all, y_pred_all)
                pr_auc = metrics.pr_auc(y_target_all, y_pred_all)
                logging.info("batch #: {}, samples processed: {}, cross entropy :  {:.4f}, cross entropy naive:  {:.4f}, relative cross entropy : {:.2f},  precision-recall AUC : {:.4f}".format(batch_count, batch_count*batch_size, cross_entropy, cross_entropy_naive, rce, pr_auc))
                t1 = time.time()
            #get the next batch of data
            X,y = next(data_gen)
            
            X_train = utils.batch_tokens_to_matrix(X,vocab, vocab_size)
            y_train = np.array(y)[:,engagement_mode]

            batch_count +=1

        except StopIteration:
            break

    #print(y_pred_all.shape, y_target_all.shape)

def main():
    
    # Set the logger
    logging_dir = args.log_dir
    engage_mode = args.engage_mode
    try:
        os.makedirs(logging_dir)
    except OSError as e:
        print("loggiing directory exists")

    log_file = 'train_MultinomialNB_' + str(engage_mode) + '.log'

    utils.set_logger(os.path.join(logging_dir, log_file))
    logging.info('model = sklearn.naive_bayes.MultinomialNB, engagement mode: reply')
    
    batch_size = args.batch_size
    data_dir = args.data_dir
    vocab_file = args.vocab_file

    logging.info('batch size = {}'.format(batch_size))


    vocab_file_path = os.path.join(data_dir, vocab_file)
    vocab_handle = bz2.open(vocab_file_path, 'r')
    vocab = pickle.load(vocab_handle)

    #train_file = "train_first_1000.tsv"
    train_file = args.train_file
    #train_file =  args.eval_file
    trainfile_path = os.path.join(data_dir, train_file)

    TR_train = tweetrecords.TweetRecords(trainfile_path,batch_size)
    data_gen_train = TR_train.tokens_target_gen()

    rng = np.random.RandomState(2020)
    model = MultinomialNB()
    logging.info('Starting Training')
    train_eval(model, data_gen_train, vocab, engage_mode, train=True)
    
    print(len(vocab))
    logging.info('Finished Training')

    with open(args.model_save_file, 'wb') as handle:
        pickle.dump(model, handle)

    logging.info('Starting Eval')

    val_file = args.eval_file
    #val_file = "train_first_1000.tsv"
    valfile_path = os.path.join(data_dir, val_file)

    TR_val = tweetrecords.TweetRecords(valfile_path,batch_size)
    data_gen_val = TR_val.tokens_target_gen()
    print(len(vocab))
    train_eval(model, data_gen_val, vocab, engage_mode, train=False)

    

    TR_train.close_file()
    TR_val.close_file()


if __name__ == "__main__":
    main()
