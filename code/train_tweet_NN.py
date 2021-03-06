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

from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import MLP_Baseline
import NCF_MLP
import utils
import tweetrecords
import metrics
import tf_idf

parser = argparse.ArgumentParser()
parser.add_argument('--engage_mode', help='tweet: 0, retweet: 1, retweet with comment: 2, like:3', default='retweet')
parser.add_argument('--batch_size', help='batch size', type=int, default=1000)

#train_first_1000
trainf = 'train_retweet_min_19.tsv'
valf = 'val_retweet_min_19.tsv'
parser.add_argument('--model', help = 'Model type: MLP or NCF', default = 'MLP')
parser.add_argument('--mode', help = 'train or eval: train_eval or eval', default = 'train_eval')

model_name_MLP = 'MLP'
model_name_NCF = 'NCF_MF_MLP'
parser.add_argument('--model_name', help = 'change the model name as needed', default = model_name_MLP)
MLP_pretrained = 'MLP_R4.pt'
NCF_pretrained = "NCF_MF_MLP.pt"
parser.add_argument('--pretrained_model', help = 'filename of pretrained model',  default = MLP_pretrained)

parser.add_argument('--vocab_file', help='vocabulary term document count file', default = 'document_freq_count_train.pickle')
parser.add_argument('--train_file', help = 'name of train file', default = trainf)
parser.add_argument('--eval_file', help = 'name of validation file', default = valf)
#parser.add_argument('--model_save_file', help = 'name of file to save trained model: train_MLP_Baseline, train_NCF_MLP', default = "../output/train_MLP_NCF.pickle")

parser.add_argument('--data_dir', help="data directory", default='../../s3mnt/data')
parser.add_argument('--output_dir', help = 'output data directory', default = "../output")
parser.add_argument('--saved_model_dir', help = 'directory to save trained models', default = "../saved_models")
parser.add_argument('--log_dir', help = 'output data directory', default = "../logs")


parser.add_argument('--log_every', help = 'log data every this number of epochss', default = 1)
parser.add_argument('--eval_every', help = 'log data every this number of batches', default = 2)
parser.add_argument('--train_epochs', help = 'Train this many number of epochs', default = 30)

parser.add_argument('--optimizer', help = 'NN Optmizer: choices: adam, rmsprop, sgd', default = 'adam')
parser.add_argument('--adam_lr', help = 'learning rate: adam', default = 0.0001)
parser.add_argument('--l2_regularization', help = 'l2_regularization parameter', default = 0.000001)
parser.add_argument('--sgd_lr', help = 'learning rate: SGD', default = 0.0001)
parser.add_argument('--sgd_momentum', help = 'momentum in SGD', default = 0.9)
parser.add_argument('--pos_weight', help = 'positive weight in BCE loss function', default = 0.22)


parser.add_argument('--layers', help = 'Layers for Baseline MLP', default = [1076,100,10])

parser.add_argument('--layers_ncf', help = 'Layers for Baseline MLP', default = [1076,350, 100])
parser.add_argument('--engaging_users', help = 'Number of engaging users (can be smaller than actual)', default = 6000)
parser.add_argument('--latent_dim_mf', help = 'embedding dim for MF', default = 100)
parser.add_argument('--latent_dim_mlp', help = 'embedding dim for MLP', default = 100)

args = parser.parse_args()

engagement_dict = {'tweet': 0, 'retweet': 1, 'retweet_comment': 2, 'like':3}

def main():
    # Set the logger
    logging_dir = args.log_dir
    engage_mode = args.engage_mode
    try:
        os.makedirs(logging_dir)
    except OSError as e:
        print("loggiing directory exists")

    log_file = args.model_name +'_' + str(engage_mode) + '.log'
    utils.set_logger(os.path.join(logging_dir, log_file))
    
    #logging.info('args parameters: %r ', args)
    for arg, value in sorted(vars(args).items()):
        logging.info("Argument %s: %r", arg, value)    
    batch_size = args.batch_size
    data_dir = args.data_dir
    vocab_file = args.vocab_file
        
    #Specify Train and Validation filepath
    train_filepath = os.path.join(data_dir, args.train_file)
    val_filepath = os.path.join(data_dir, args.eval_file)

    #Instantiating Val tweet records
    TR_val = tweetrecords.TweetRecords(val_filepath, batch_size, embed_tokens_next=True)
    logging.info("loading unique_id_token embeddings dict")
    fhandle = bz2.BZ2File(os.path.join(data_dir, 'val_retweet_min_19_token_embeds_mean.pickle'), 'r')
    TR_val.unique_id_tokenembedsdict = pickle.load(fhandle)
    fhandle.close()

    rng = np.random.RandomState(2020)
   
    if args.model == 'MLP':
        logging.info('model = Multi layer NN with BERT embeddings and other features as input, engagement mode: Retweet')
        logging.info('Does not use UserID embeddings')
        logging.info('Using BERT embeddings values of tweet tokens and one hot encoding of other features')
        engine = MLP_Baseline.MLP_BaselineEngine(args)
        
    elif args.model == 'NCF':
        logging.info('model  = Neural Collaborative Filtering, engagement, Optimizer: Adam, mode: Retweet')
        logging.info('Uses  user id embeddings')
        engine = NCF_MLP.NCF_Engine(args)
        args.model_name = model_name_NCF
        args.pretrained_model = NCF_pretrained

    if args.mode == 'train_eval':
        #Instantiating Train tweet records
        logging.info("Starting Training")
        logging.info('training set: '.format(args.train_file))
        TR_train = tweetrecords.TweetRecords(train_filepath, batch_size, embed_tokens_next=True)
        print("loading unique_id_token embeddings dict")

        fhandle = bz2.BZ2File(os.path.join(data_dir, 'train_retweet_min_19_token_embeds_mean.pickle', 'r'))

        TR_train.unique_id_tokenembedsdict = pickle.load(fhandle)
        fhandle.close()

        for i in range(1):
            #args.l2_regularization = 1.0/(10**(i+1))
            #engine = NCF_MLP.NCF_Engine(args)
            for epoch in range(args.train_epochs):
                print('-' * 80)
                print('Epoch {} starts !  l2reg :{}'.format(epoch, args.l2_regularization))
                print('-' * 80)
                
                engine.train_an_epoch(TR_train, args, epoch, mode='train')
                if (epoch % args.eval_every ==0):
                    engine.save_model(args, epoch)
                    engine.train_an_epoch(TR_val, args, epoch, mode='eval')
        TR_train.close_file()
    
        logging.info('Finished Training')
    elif args.mode == 'eval':
        logging.info("Starting Eval")
        saved_model_path = os.path.join(args.saved_model_dir  ,  args.pretrained_model)
        engine.load_model(saved_model_path)
        engine.train_an_epoch(TR_val, args, 0, mode='eval')
        logging.info('Finished Eval')

    TR_val.close_file()

if __name__ == "__main__":
    main()
