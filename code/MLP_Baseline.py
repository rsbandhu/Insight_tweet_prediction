import torch
import torch.nn as nn
#from transformers import BertModel

from engine import Engine

class MLP_Baseline(nn.Module):
    def __init__(self, args):
        super(MLP_Baseline, self).__init__()
        #self.BERT_multilingual = BertModel.from_pretrained( 'bert-base-multilingual-cased')
        self.fc_layers = nn.ModuleList()
        
        #add the FC layers to the list
        for _, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            print(in_size, out_size)
            self.fc_layers.append(nn.Linear(in_size, out_size))
        #final output layer
        self.out_layer = nn.Linear(args.layers[-1], 1) 
        self.logistic = nn.Sigmoid()
        self.model_name = 'MLP_Baseline'

    def forward(self, token_embeds_mean, features, engaging_user=None, engaged_user=None):
        
        #concat BERT embeddings of tokens with tweet features
        X = torch.cat((token_embeds_mean, features), axis = 1)
        
        for i in range(len(self.fc_layers)):
            X = self.fc_layers[i](X)
            X = nn.ReLU()(X)

        logits = torch.squeeze(self.out_layer(X))
        #print(logits.shape)
        #engage_prob = torch.squeeze(self.logistic(logits))

        return logits


class MLP_BaselineEngine(Engine):
    """Engine for training & evaluating Baseline MLP (NN) model"""
    def __init__(self, args):
        
        self.model = MLP_Baseline(args)
        
        super(MLP_BaselineEngine, self).__init__(args)
        self.model.to(self.device)

        print(self.model)

        #if config['pretrain']:
        #    self.model.load_pretrain_weights()
