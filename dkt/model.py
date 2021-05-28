import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy
import math

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel    
except:
    from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel    




class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args
        self.d = vars(self.args)        # Namespace를 dictionary 형태로
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_features = []
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // 3))

        #self.embedding_classification = nn.Embedding(self.args.n_class + 1, self.hidden_dim//3)
        #self.embedding_paperNum = nn.Embedding(self.args.n_paper + 1, self.hidden_dim//3)
        #self.embedding_problemNum = nn.Embedding(self.args.n_problem + 1, self.hidden_dim//3)
        #self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim//3)

        # embedding combination projection
        # +1은 interaction
        self.comb_proj = nn.Linear((self.hidden_dim//3)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        _, mask, interaction, _ = input[-4:]

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            embedding_feature = _embedding_feature.to(self.args.device)
            value = embedding_feature(_input)
            embed_features.append(value)
        #embed_classification = self.embedding_classification(classification)
        #embed_paperNum = self.embedding_paperNum(paperNum)
        #embed_problemNum = self.embedding_problemNum(problemNum)
        #embed_tag = self.embedding_tag(tag)

        embed_features = [embed_interaction] + embed_features

        embed = torch.cat(embed_features, 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds


class LSTMATTN(nn.Module):

    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding 
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_features = []
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // 3))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

        self.lstm = nn.LSTM(self.hidden_dim,
                            self.hidden_dim,
                            self.n_layers,
                            batch_first=True)
        
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)            
    
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.n_layers,
            batch_size,
            self.hidden_dim)
        c = c.to(self.device)

        return (h, c)

    def forward(self, input):

        _, mask, interaction, _ = input[-4:]

        batch_size = interaction.size(0)

        # Embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            embedding_feature = _embedding_feature.to(self.args.device)
            value = embedding_feature(_input)
            embed_features.append(value)

        embed_features = [embed_interaction] + embed_features

        embed = torch.cat(embed_features, 2)

        X = self.comb_proj(embed)

        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(X, hidden)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
                
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers
        
        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)        
        sequence_output = encoded_layers[-1]
        
        out = self.fc(sequence_output)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class Bert(nn.Module):

    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_features = []
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // 3))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//3)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len          
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()


    def forward(self, input):
        _, mask, interaction, _ = input[-4:]

        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            embedding_feature = _embedding_feature.to(self.args.device)
            value = embedding_feature(_input)
            embed_features.append(value)

        embed_features = [embed_interaction] + embed_features

        embed = torch.cat(embed_features, 2)


        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)
        preds = self.activation(out).view(batch_size, -1)

        return preds