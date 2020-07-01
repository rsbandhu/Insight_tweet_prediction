import torch
import torch.nn as nn
#from transformers import BertModel

from engine import Engine

class NCF(nn.Module):
    def __init__(self, args):
        super(NCF, self).__init__()
        
        self.fc_layers = nn.ModuleList()
        
        self.num_users = args.engaging_users
        self.latent_dim_mf = args.latent_dim_mf
        self.latent_dim_mlp = args.latent_dim_mlp

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)


        #add the FC layers to the list
        for _, (in_size, out_size) in enumerate(zip(args.layers_ncf[:-1], args.layers_ncf[1:])):
            print(in_size, out_size)
            self.fc_layers.append(nn.Linear(in_size, out_size))

        #MLP Layers
        self.mlp1 = nn.Linear(args.layers_ncf[-1] + self.latent_dim_mlp, self.latent_dim_mlp)
        self.mlp2 = nn.Linear(self.latent_dim_mlp, 50)

        #final output layers
        self.out_layer1 = nn.Linear(50 + self.latent_dim_mf, 20) 
        self.out_layer2 = nn.Linear(20 , 1) 

        self.logistic = nn.Sigmoid()
        self.model_name = 'NCF_MF_MLP'

    def forward(self, token_embeds_mean, features, engaging_user, engaged_user=None):
        
        #concat BERT embeddings of tokens with tweet features
        X = torch.cat((token_embeds_mean, features), axis = 1) #shape = (batch_size, 1076)

        #Reduce the word-embeddings and features to output size = (batch_size, 100)
        for i in range(2):
            X = self.fc_layers[i](X)
            X = nn.ReLU()(X)
        
        #Embed engaging user ID (one hot) and tweet features
        user_features_embed_mlp = self.embedding_user_mlp(engaging_user)
        user_features_embed_mf = self.embedding_user_mf(engaging_user)

        #Kronecker product of token embeddings and  (userID + feature) embeddings
        X_mf =torch.mul(user_features_embed_mf, X)

        # concat token embeddings with (userID + feature) embeddings for MLP
        X_mlp = torch.cat([X, user_features_embed_mlp], dim=-1)  

        #Process through MLP layers
        X_mlp = self.mlp1(X_mlp)
        X_mlp = nn.ReLU()(X_mlp)
        X_mlp = self.mlp2(X_mlp)
        X_mlp = nn.ReLU()(X_mlp)

        # concat output of MF and MLP
        X_mlp_mf = torch.cat([X_mlp, X_mf], dim=-1)

        #Final FC layers
        X_mlp_mf = self.out_layer1(X_mlp_mf)
        X_mlp_mf = nn.ReLU()(X_mlp_mf)
        
        logits = self.out_layer2(X_mlp_mf)
        engage_prob = torch.squeeze(self.logistic(logits))
        
        return engage_prob


class NCF_Engine(Engine):
    """Engine for training & evaluating Baseline MLP (NN) model"""
    def __init__(self, args):
        
        self.model = NCF(args)
        
        super(NCF_Engine, self).__init__(args)
        self.model.to(self.device)

        print(self.model)
