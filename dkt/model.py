import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import pandas as pd
import copy
import math
import re
import os
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pycaret.classification import *
from pycaret.utils import check_metric
import random

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
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        # embedding combination projection
        # +1은 interaction
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

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
            value = _embedding_feature(_input)
            embed_features.append(value)

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
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

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
            is_decoder=self.args.is_decoder
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
            value = _embedding_feature(_input)
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
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            intermediate_size=self.hidden_dim*2,
            max_position_embeddings=self.args.max_seq_len,
            hidden_dropout_prob=self.args.drop_out,
            attention_probs_dropout_prob=self.args.drop_out,
            is_decoder=self.args.is_decoder
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
            value = _embedding_feature(_input)
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
    

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self,ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))

class LastQuery(nn.Module):
    def __init__(self, args):
        super(LastQuery, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        
        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.args.dim_div)

#         self.embedding_position = nn.Embedding(self.args.max_seq_len, self.hidden_dim)
    
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)
        
        # Encoder
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.args.n_heads)
        self.mask = None # last query에서는 필요가 없지만 수정을 고려하여서 넣어둠
        self.ffn = Feed_Forward_block(self.hidden_dim)      

        self.ln1 = nn.LayerNorm(self.hidden_dim)
        self.ln2 = nn.LayerNorm(self.hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            self.hidden_dim,
            self.hidden_dim,
            self.args.n_layers,
            batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()
        
        # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixup Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

        
        
    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*ln.*|.*bn.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param          
            elif re.match(r'encoder.*ffn.*weight$|encoder.*attn.out_proj.weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]

        self.load_state_dict(temp_state_dict)
        
    # lastquery는 2D mask가 필요함
    def get_mask(self, seq_len, mask, batch_size):
        new_mask = torch.zeros_like(mask)
        new_mask[mask == 0] = 1
        new_mask[mask != 0] = 0
        mask = new_mask
    
        # batchsize * n_head 수만큼 각 mask를 반복하여 증가시킨다
        mask = mask.repeat(1, self.args.n_heads).view(batch_size*self.args.n_heads, -1, seq_len)
        return mask.masked_fill(mask==1, float('-inf'))

    def get_pos(self, seq_len):
        # use sine positional embeddinds
        return torch.arange(seq_len).unsqueeze(0)
 
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.device)

        c = torch.zeros(
            self.args.n_layers,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.device)

        return (h, c)


    def forward(self, input):
        _, mask, interaction, index = input[-4:]
        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)
    
        embed_features = [embed_interaction] + embed_features
        embed = torch.cat(embed_features, 2)

        embed = self.comb_proj(embed)

        # Positional Embedding
        # last query에서는 positional embedding을 하지 않음
#         position = self.get_pos(seq_len).to('cuda')
#         embed_pos = self.embedding_position(position)
#         embed = embed + embed_pos

        ####################### ENCODER #####################
        q = self.query(embed)[:, -1:, :].permute(1, 0, 2)
        k = self.key(embed).permute(1, 0, 2)
        v = self.value(embed).permute(1, 0, 2)

        ## attention
        # last query only
        self.mask = self.get_mask(seq_len, mask, batch_size).to(self.device)
        out, _ = self.attn(q, k, v, attn_mask=self.mask)
        
        ## residual + layer norm
        out = out.permute(1, 0, 2)
        out = embed + out
        out = self.ln1(out)

        ## feed forward network
        out = self.ffn(out)

        ## residual + layer norm
        out = embed + out
        out = self.ln2(out)

        ###################### LSTM #####################
        hidden = self.init_hidden(batch_size)
        out, hidden = self.lstm(out, hidden)

        ###################### DNN #####################
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class Saint(nn.Module):
    
    def __init__(self, args):
        super(Saint, self).__init__()
        self.args = args
        self.device = args.device

        self.hidden_dim = self.args.hidden_dim
        # self.dropout = self.args.dropout
        self.dropout = 0.
        
        ### Embedding 
        # ENCODER embedding
        # test, question, tag
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))
        
        # encoder combination projection
        self.enc_comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*len(self.args.n_embedding_layers), self.hidden_dim)

        # DECODER embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.args.dim_div)
        
        # decoder combination projection
        self.dec_comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        self.pos_decoder = PositionalEncoding(self.hidden_dim, self.dropout, self.args.max_seq_len)
        

        self.transformer = nn.Transformer(
            d_model=self.hidden_dim, 
            nhead=self.args.n_heads,
            num_encoder_layers=self.args.n_layers, 
            num_decoder_layers=self.args.n_layers, 
            dim_feedforward=self.hidden_dim, 
            dropout=self.dropout, 
            activation='relu')

        self.fc = nn.Linear(self.hidden_dim, 1)
        self.activation = nn.Sigmoid()

        self.enc_mask = None
        self.dec_mask = None
        self.enc_dec_mask = None
    
    def get_mask(self, seq_len):
        mask = torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1))

        return mask.masked_fill(mask==1, float('-inf'))

    def forward(self, input):
        _, mask, interaction, index = input[-4:]

        batch_size = interaction.size(0)
        seq_len = interaction.size(1)

        # 신나는 embedding
        # ENCODER
        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)

        embed_enc = torch.cat(embed_features, 2)

        embed_enc = self.enc_comb_proj(embed_enc)
        
        # DECODER     
        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            value = _embedding_feature(_input)
            embed_features.append(value)

        embed_interaction = self.embedding_interaction(interaction)
        embed_features = [embed_interaction] + embed_features
        embed_dec = torch.cat(embed_features, 2)

        embed_dec = self.dec_comb_proj(embed_dec)

        # ATTENTION MASK 생성
        # encoder하고 decoder의 mask는 가로 세로 길이가 모두 동일하여
        # 사실 이렇게 3개로 나눌 필요가 없다
        if self.enc_mask is None or self.enc_mask.size(0) != seq_len:
            self.enc_mask = self.get_mask(seq_len).to(self.device)
            
        if self.dec_mask is None or self.dec_mask.size(0) != seq_len:
            self.dec_mask = self.get_mask(seq_len).to(self.device)
            
        if self.enc_dec_mask is None or self.enc_dec_mask.size(0) != seq_len:
            self.enc_dec_mask = self.get_mask(seq_len).to(self.device)
            
  
        embed_enc = embed_enc.permute(1, 0, 2)
        embed_dec = embed_dec.permute(1, 0, 2)
        
        # Positional encoding
        embed_enc = self.pos_encoder(embed_enc)
        embed_dec = self.pos_decoder(embed_dec)
        
        out = self.transformer(embed_enc, embed_dec,
                               src_mask=self.enc_mask,
                               tgt_mask=self.dec_mask,
                               memory_mask=self.enc_dec_mask)

        out = out.permute(1, 0, 2)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out)

        preds = self.activation(out).view(batch_size, -1)

        return preds


class TfixupBert(nn.Module):
    def __init__(self, args):
        super(TfixupBert, self).__init__()
        self.args = args
        self.device = args.device

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding 
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim//self.args.dim_div)
        self.embedding_features = nn.ModuleList([])
        for value in self.args.n_embedding_layers:
            self.embedding_features.append(nn.Embedding(value + 1, self.hidden_dim // self.args.dim_div))

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim//self.args.dim_div)*(len(self.args.n_embedding_layers)+1), self.hidden_dim)

        # Bert config
        self.config = BertConfig( 
            3, # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            intermediate_size=self.hidden_dim*2,
            max_position_embeddings=self.args.max_seq_len,
            hidden_dropout_prob=self.args.drop_out,
            attention_probs_dropout_prob=self.args.drop_out,
            is_decoder=self.args.is_decoder         
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config)  

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)
       
        self.activation = nn.Sigmoid()
        
        # T-Fixup
        if self.args.Tfixup:

            # 초기화 (Initialization)
            self.tfixup_initialization()
            print("T-Fixupbb Initialization Done")

            # 스케일링 (Scaling)
            self.tfixup_scaling()
            print(f"T-Fixup Scaling Done")

    def tfixup_initialization(self):
        # 우리는 padding idx의 경우 모두 0으로 통일한다
        padding_idx = 0

        for name, param in self.named_parameters():
            if re.match(r'^embedding*', name):
                nn.init.normal_(param, mean=0, std=param.shape[1] ** -0.5)
                nn.init.constant_(param[padding_idx], 0)
            elif re.match(r'.*Norm.*', name):
                continue
            elif re.match(r'.*weight*', name):
                # nn.init.xavier_uniform_(param)
                nn.init.xavier_normal_(param)


    def tfixup_scaling(self):
        temp_state_dict = {}

        # 특정 layer들의 값을 스케일링한다
        for name, param in self.named_parameters():

            # TODO: 모델 내부의 module 이름이 달라지면 직접 수정해서
            #       module이 scaling 될 수 있도록 변경해주자
            # print(name)

            if re.match(r'^embedding*', name):
                temp_state_dict[name] = (9 * self.args.n_layers) ** (-1 / 4) * param   
            elif re.match(r'.*Norm.*', name):
                continue
            elif re.match(r'encoder.*dense.*weight$|encoder.*attention.output.*weight$', name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * param
            elif re.match(r"encoder.*value.weight$", name):
                temp_state_dict[name] = (0.67 * (self.args.n_layers) ** (-1 / 4)) * (param * (2**0.5))

        # 나머지 layer는 원래 값 그대로 넣는다
        for name in self.state_dict():
            if name not in temp_state_dict:
                temp_state_dict[name] = self.state_dict()[name]
                
        self.load_state_dict(temp_state_dict)


    def forward(self, input):
        _, mask, interaction, index = input[-4:]
        batch_size = interaction.size(0)

        # 신나는 embedding
        embed_interaction = self.embedding_interaction(interaction)
        embed_features = []
        for _input, _embedding_feature in zip(input[:-4], self.embedding_features):
            value = _embedding_feature(_input)
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


class TabNet(nn.Module):
    def __init__(self, args):
        super(TabNet, self).__init__()
        self.args = args
        self.device = args.device
        
        self.optimizer = self.get_optimizer(self.args.tabnet_optimizer)
        self.scheduler = self.get_scheduler(self.args.tabnet_scheduler)
        
        if self.args.tabnet_pretrain:
            self.unsupervised_model = TabNetPretrainer(
                optimizer_fn=self.optimizer,
                optimizer_params=dict(lr=self.args.tabnet_lr),
                mask_type=self.args.tabnet_mask_type, # "sparsemax",
                seed=self.args.seed,
                saving_path=os.path.join(self.args.model_dir, self.args.model_name),
                )
            
        self.clf = TabNetClassifier(
            optimizer_fn=self.optimizer,
            optimizer_params=dict(lr=self.args.tabnet_lr),
            scheduler_params={"step_size":self.args.tabnet_n_step, # how to use learning rate scheduler
                              "gamma":self.args.tabnet_gamma},
            scheduler_fn=self.scheduler,
            mask_type=self.args.tabnet_mask_type, # This will be overwritten if using pretrain model
            seed=self.args.seed,
            )
        
    def get_optimizer(self, opt):
        if self.args.tabnet_optimizer == 'adam':
            optimizer = torch.optim.Adam
        if self.args.tabnet_optimizer == 'adamW':
            optimizer = torch.optim.AdamW
        if self.args.tabnet_optimizer == 'SGD':
            optimizer = torch.optim.SGD
        # 모든 parameter들의 grad값을 0으로 초기화
        return optimizer
    def get_scheduler(self, sch):
        if self.args.tabnet_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
        elif self.args.tabnet_scheduler == 'steplr':
            scheduler = torch.optim.lr_scheduler.StepLR
        return scheduler
    
    def forward(self):
        if self.args.tabnet_pretrain:
            return self.unsupervised_model, self.clf
        return self.clf


class LGBM(nn.Module):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
    def fit(self, X_train, y_train, X_valid, y_valid, FEATS, categorical_features=[],numeric_features=[],seed=47):

        X_trn = X_train.merge(y_train, on=X_train.index)
        X_val = X_valid.merge(y_valid, on=X_valid.index)

        random.seed(seed)
        settings = setup(data=X_trn[FEATS], target='answerCode', categorical_features=categorical_features, numeric_features=numeric_features, silent=True)
        
        lgbm = create_model('lightgbm', sort='AUC')
        tuned_lgbm = tune_model(lgbm, optimize='AUC', fold=10)
        final_lgbm = finalize_model(tuned_lgbm)

        log = []
        prediction = predict_model(final_lgbm, data=X_val[FEATS], raw_score=True)
        log.append(f"{check_metric(prediction['answerCode'], prediction['Label'], metric='Accuracy')} ,{check_metric(prediction['answerCode'], prediction['Score_1'], metric='AUC')}")
        return final_lgbm, log