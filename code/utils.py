import numpy as np
import time
import datetime
from datetime import timezone
import os
import logging
import bz2
import pickle

import torch

def create_onehot(buckets, index):
    '''
    Create a one hot vector
    args:
    buckets: dimension of the vector
    index: location where it is 1
    returns:
    onehot vector
    '''
    #y_onehot = torch.zeros(buckets)
    y_onehot = [0]*buckets
    y_onehot[index] = 1
    return y_onehot

def create_hist_bucket_index(bucket_boundaries, value):
    '''
    Given a list of values, bucket_boundaries= [a1,a2,an], and a value x
    find the index i such that  x < bucket_boundaries[i]
    args:
    bucket_boundaries: list of numbers, [a1,a2,an]
    first bucket, [-inf, a1), index = 0
    second bucket, [a1, a2), index = 1
    last bucket, [an, inf), index = n
    value: 
    
    returns:
    bucket_count: total number of buckets (dimension of onehot vector)
    bucket_id: index of value in the bucket
    '''
    
    bucket_count = len(bucket_boundaries) +1  # n+1
    for i in range(bucket_count-1):  # 0 through (n-1)
        if (value < bucket_boundaries[i]):
            bucket_id = i
            break
    if (value >= bucket_boundaries[-1]):
        bucket_id = bucket_count-1  # index = n
    return (bucket_count, bucket_id)



def extract_datetime(unix_time):
    time_stamp = datetime.datetime.fromtimestamp(int(unix_time),tz= timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    #print(datetime.datetime.fromtimestamp(int(unix_time)))
    return time_stamp


def extract_follower_following_quantiles(filepath):
    '''
    returns the quantile values of the follower and following counts of each user
    Asssumes 32 quantiles, i/32, i = 1,2,... 31
    Inputs:
    filepath: pickled bz2 file that contains dict, 
    key: userid, value:[follower count, following count]
    Returns:
    2D numpy array (shape: (31,2)) containing the quantile values for each follow type

    '''
    ff_handle = bz2.BZ2File(filepath)
    ff_dict = pickle.load(ff_handle)

    ff_counts = np.array(list(ff_dict.values()))
    quantiles = np.array([i/32.0 for i in range(1,32)])
    ff_quantiles = np.quantile(ff_counts, quantiles, axis=0)

    #print("quantiles shape = ",ff_quantiles.shape)
    #print(ff_quantiles)

    return ff_quantiles
    

def batch_tokens_to_matrix(tokens_batch, vocabulary, vocab_size):
    '''
    generates a token counts for a batch of token
    args: 
    tokens_batch: batch of tokens. Each items in the batch can have different number of tokens
    vocabulary: vocab dict containing tokens to index mapping
    returns:
    matrix of shape (batch_size, voacb_size), each element containing token count for that index
    '''
    batch_size = len(tokens_batch)
    #print("before tokens to matrix: ", batch_size, vocab_size, len(vocabulary))
    zero_term_dict = {key:0 for key in vocabulary.keys()}
    '''
    Test if vocabulary is unchanged:
    
    vocab_keys = [k for k in vocabulary.keys()]
    print([vocabulary[k] for k in vocab_keys[-10:]])
    '''
    ret = []
    
    for recnum, tokens in enumerate(tokens_batch):
        tokens_dict = raw_count(tokens, zero_term_dict)
        #print(tokens)
        token_list = list(tokens_dict.values())
        #print(len(token_list))
        ret.append(token_list)
        
    #print("inside tokens to matrix: ", batch_size, vocab_size, len(vocabulary))
    ret = np.array(ret)
    #print("shape of ret: ", ret.shape)
    #print([vocabulary[k] for k in vocab_keys[-10:]])
    return ret

def extract_year_month_day_hour_min_weekday(unix_time):

    # format of time_stamp is :  2020-02-11 00:00:10
    if len(unix_time) == 0:
        return None
    else:
        time_stamp = datetime.datetime.fromtimestamp( int(unix_time), tz=timezone.utc )
        #time_stamp = time_stamp.split()
        #year_month_day = time_stamp[0].split("-")
        year = time_stamp.year
        month = time_stamp.month
        day = time_stamp.day
        hr = time_stamp.hour
        minute = time_stamp.minute
        weekday = time_stamp.weekday()
        return (year, month, day, hr, minute, weekday)

def get_current_local_time():
    time_stamp = int(time.time())
    time_stamp = datetime.datetime.fromtimestamp(time_stamp)
    year = str(time_stamp.year)
    month = str(time_stamp.month)
    day = str(time_stamp.day)
    hr = str(time_stamp.hour)
    minute = str(time_stamp.minute)

    return year+'_'+month+'_'+day+'_'+hr+'_'+minute



def raw_count(tokens_list, zero_term_dict):
        tokens_dict = {key:0 for key in zero_term_dict.keys()}
        for item in tokens_list:
            if (item != 0):
                tokens_dict[item] += 1
        
        return tokens_dict

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) #20 for INFO, 40 for ERROR, 10 for DEBUG

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(model_dir,
                            map_location=lambda storage, loc: storage.cuda(device=device_id))  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, args):
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr= args.sgd_lr,
                                    momentum=args.sgd_momentum,
                                    weight_decay=args.l2_regularization)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                    lr=args.adam_lr,
                                    weight_decay=args.l2_regularization)
    return optimizer