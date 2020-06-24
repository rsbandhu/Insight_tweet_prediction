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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data

import utils
import tweetrecords
import metrics

class Engine(object):

    def __init__(self, args):
        self.opt = utils.use_optimizer(self.model, args)
        self.device = self.set_device()
        w_pos = torch.tensor([args.pos_weight]).to(self.device)
        self.loss = nn.BCEWithLogitsLoss(pos_weight= w_pos)

    def train_single_batch(self, tokens_embed_batch, features_batch, y_t, engaging_user=None, engaged_user=None):
            assert hasattr(self, 'model'), 'Please specify the exact model !'
            engagement_mode = 1
            tokens_embed_batch = torch.as_tensor(np.array(tokens_embed_batch), dtype=torch.float).squeeze()
            #print(tokens_embed_batch.shape, torch.max(tokens_embed_batch), torch.min(tokens_embed_batch))
            #tokens_embed_batch.requires_grad = False
            tokens_embed_batch = tokens_embed_batch.to(self.device)

            y_train = torch.as_tensor(y_t, dtype=torch.float)
            y_train = y_train.to(self.device)

            features_batch = torch.as_tensor(features_batch, dtype=torch.float)
            features_batch = features_batch.to(self.device)
          
            if engaging_user is not None:
                engaging_user = torch.as_tensor(engaging_user, dtype=torch.long)
                engaging_user = engaging_user.to(self.device)

            #print("shapes::  tokens: {}, features: {}, y_targets: {}, engaging_user: {}".format(tokens_embed_batch.shape, features_batch.shape, y_train.shape, engaging_user.shape))
            #print(torch.max(tokens_embed_batch), torch.min(tokens_embed_batch))
            
            self.opt.zero_grad()
            y_pred = self.model(tokens_embed_batch, features_batch, engaging_user, engaged_user)
            #print(torch.max(y_pred), torch.min(y_pred))

            loss = self.loss(y_pred, y_train)
            #print("shape of loss", loss.shape)
            loss.backward()
            self.opt.step()
            loss = loss.item()

            return loss, y_pred.detach().cpu().numpy()

    def eval_single_batch(self, tokens_embed_batch, features_batch, y_t, engaging_user=None, engaged_user=None):
            assert hasattr(self, 'model'), 'Please specify the exact model !'
            engagement_mode = 1
            tokens_embed_batch = torch.as_tensor(np.array(tokens_embed_batch), dtype=torch.float).squeeze().to(self.device)
            tokens_embed_batch.requires_grad = False
         
            y_train = torch.as_tensor(y_t, dtype=torch.float).to(self.device)

            features_batch = torch.as_tensor(features_batch, dtype=torch.float).to(self.device)

            if engaging_user is not None:
                engaging_user = torch.as_tensor(engaging_user, dtype=torch.long).to(self.device)
            #print("shapes::  tokens: {}, features: {}, y_targets: {}".format(tokens_batch.shape, features_batch.shape, y_train.shape))   
            with torch.no_grad():
                y_pred = self.model(tokens_embed_batch, features_batch, engaging_user, engaged_user)
                loss = self.loss(y_pred, y_train)
                loss = loss.item()

            return loss, y_pred.detach().cpu().numpy()

    def train_an_epoch(self,TR, args, epoch_id, mode = 'train'):

        t1 = time.time()
        #set data generator of tweet records: train or eval
        TR.get_file_handle()
        data_gen = TR.tokenembeds_target_features_gen()

        #set model to train or eval mode
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval() 

        if (args.engage_mode == 'retweet'):
            engagement_mode = 1 
        #get the first batch of X and y data
        tokens_batch,yt_batch, features_batch, engaging_userid_hash_batch = next(data_gen)
        batch_size = len(tokens_batch)
        #placeholders for storing the targets and predictions for epoch
        y_target_epoch = np.array([])
        y_pred_epoch = np.array([])
        batch_count = 1  #counts the number of batches
        total_loss = 0
        engaging_user=None
        engaged_user=None
        while tokens_batch:
            try:
                yt_batch = np.array(yt_batch)[:, engagement_mode] # 1  for engagement mode = retweet
                #print("number of positive sample :  ",np.sum(yt_batch))
                if mode == 'train':
                    loss, y_pred = self.train_single_batch(tokens_batch, features_batch, yt_batch, engaging_userid_hash_batch, engaged_user)
                else:
                    loss, y_pred = self.eval_single_batch(tokens_batch, features_batch, yt_batch, engaging_userid_hash_batch, engaged_user)
                total_loss += loss
                #print('[Training Epoch: {}] total_loss: {:.5f}, Loss: {:.5f}'.format(epoch_id, total_loss, loss))
                #y_pred = y_pred[:, 1]
                
                #logging.info("epoch_id: {}, y_pred_max: {}, y_pred_90p: {}, y_pred_10p: {}, y_pred_min: {} ".format(epoch_id, np.max(y_pred), np.quantile(y_pred, 0.9), np.quantile(y_pred, 0.1), np.min(y_pred)))
                
                if (np.isnan(y_pred).any()):
                    break

                if y_pred_epoch.shape[0] > 0: #if this is not first batch
                    y_pred_epoch = np.append(y_pred_epoch, y_pred, axis=0) 
                else:
                    y_pred_epoch = y_pred #for first batch
                    
                y_target_epoch = np.append(y_target_epoch, yt_batch)
                #get the next batch of tokens, features and targets
                tokens_batch,yt_batch, features_batch, engaging_userid_hash_batch = next(data_gen)
                #print(batch_count)
                batch_count +=1

            except StopIteration:
                break

        #Calculate metrics
        t2 = time.time()
        print("time taken to {} :  for epoch {} : {}".format(mode, epoch_id, t2 - t1))
        if (epoch_id%args.log_every == 0):
            y_prob_epoch = metrics.logits_to_prob(y_pred_epoch)
            rce, cross_entropy, cross_entropy_naive = metrics.relative_cross_entropy(y_target_epoch, y_prob_epoch)
            pr_auc = metrics.pr_auc(y_target_epoch, y_prob_epoch)
            logging.info("{} metrics: epoch : {}, samples processed: {}, cross entropy :  {:.5f}, cross entropy naive:  {:.5f}, relative cross entropy : {:.3f},  precision-recall AUC : {:.4f}".format(mode, epoch_id, batch_count*batch_size, cross_entropy, cross_entropy_naive, rce, pr_auc))
        TR.close_file()

    def save_model(self, args, epoch_id):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        saved_model_dir = args.saved_model_dir
        model_name = self.model.model_name
        localtime = utils.get_current_local_time()
        filename = model_name+'_'+str(epoch_id)+'_'+localtime+'.pt'
        model_dir = os.path.join(saved_model_dir, filename)
        torch.save(self.model.state_dict(), model_dir)

    def set_device(self):
            #set deivce for computing NN
            if torch.cuda.is_available():
                dev = 'cuda:0'
                device = torch.device(dev)
                #torch.cuda.set_device(device)
            else:
                device = torch.device('cpu')
            return device


if __name__ == "__main__":
    main()
