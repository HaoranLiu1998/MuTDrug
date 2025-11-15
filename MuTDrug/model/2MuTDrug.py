#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import random
from tqdm import tqdm
import pickle
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
from torch_geometric.nn import GATConv,GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch, Data
from torch.cuda.amp import GradScaler, autocast
from torch_scatter import scatter_mean,scatter_add
from kan import KANLayer
import math
import os
import re
import bisect
import numpy as np
import time
import gc
from collections import defaultdict
import pandas as pd
import ast


# In[2]:


def get_model_size(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"number of parameters: {param_size}")
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024 ** 2)
    return total_size


# In[3]:


#Set device
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False
    
print("Device =", device)


# In[4]:


# USE_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 1

# if USE_CUDA:
#     device = torch.device("cuda:1")
#     cuda = True
# else:
#     device = torch.device("cpu")
#     cuda = False

# print("Device =", device)


# In[5]:


#Set time
def time_elapsed(start_time):
    elapsed = time.time() - start_time  
    hours = int(elapsed/3600)           
    minutes = int(int(elapsed/60)%60)   
    seconds = int(elapsed%60)           
    
    return hours, minutes, seconds


# In[6]:


#mRNAEncoder submodule
def sinusoidal_position_embedding(device, batch_size = 1, nums_head = 8, max_len = 6400, output_dim = 1024):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  
    theta = torch.pow(10000, -2 * ids / output_dim)

    # (max_len, output_dim//2)
    embeddings = position * theta  

    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))

    # (bs, head, max_len, output_dim)

    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings

def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]

    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(q.device, batch_size, nums_head, max_len, output_dim)


    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1) 
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)
    q = q * cos_pos + q2 * sin_pos

    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    k = k * cos_pos + k2 * sin_pos

    return q, k

def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)

    if use_RoPE:
        q, k = RoPE(q, k)

    d_k = k.size()[-1]

    att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
    att_logits /= math.sqrt(d_k)

    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e9)

    att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)

    if dropout is not None:
        att_scores = dropout(att_scores)

    # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
    return torch.matmul(att_scores, v), att_scores

# q = torch.randn((8, 12, 10, 32))
# k = torch.randn((8, 12, 10, 32))
# v = torch.randn((8, 12, 10, 32))

# q = q.to(device)
# k = k.to(device)
# v = v.to(device)

# res, att_scores = attention(q, k, v, mask=None, dropout=None, use_RoPE=True)


#     # (bs, head, seq_len, dk),  (bs, head, seq_len, seq_len)
# print(res.shape, att_scores.shape)


# In[7]:


class mRNAEncoder (nn.Module):
    def __init__(self):
        super(mRNAEncoder, self).__init__()
        self.Conv_1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=1)
        self.Conv_2 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=3)
        self.Conv_3 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=3)
        self.Conv_4 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1, stride=3)
        self.Conv_5 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=3)
        self.Conv_6 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=9, stride=3,padding=3)
        #ninp=1024, nhead=8, nhid=1024, nlayers=3, dropout=0.2
        d_model = 4*3+4*3+64+128
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=1024, dropout=0.2, batch_first=True)
        self.mRNA_transformer_encoder = TransformerEncoder(copy.deepcopy(encoder_layer), num_layers=4)
        self.outlinear = nn.Linear(4*3+4*3+64+128,1024)

        
        
    def forward(self,mRNA_seq):
        mRNA_seq = mRNA_seq[:((mRNA_seq.size()[0] - 1) // 3 * 3)].reshape(1,-1).float()
        out_1 = self.Conv_1(mRNA_seq)
        out_1_0 = out_1[0,:].reshape(-1,3).t()
        out_1_1 = out_1[1,:].reshape(-1,3).t()
        out_1_2 = out_1[2,:].reshape(-1,3).t()
        out_1_3 = out_1[3,:].reshape(-1,3).t()
        out_2 = self.Conv_2(mRNA_seq)
        out_3 = self.Conv_3(mRNA_seq[:,1:])
        out_4 = self.Conv_4(mRNA_seq[:,2:])        
        out_5 = self.Conv_5(mRNA_seq)
        out_6 = self.Conv_6(mRNA_seq)
        #168264
        cnn_out = torch.cat([out_1_0, out_1_1, out_1_2, out_1_3, out_2, out_3, out_4, out_5, out_6], 0)
        fcnn_out = cnn_out.t()
        RoPE_in = fcnn_out.reshape(1,1,-1,4*3+4*3+64+128)
        RoPE_out, att_scores = attention(RoPE_in, RoPE_in, RoPE_in, mask=None, dropout=None, use_RoPE=True)
        RoPE_out = RoPE_out.squeeze(0)##(batch,seq_len,ninp)
        
        transf_out = self.mRNA_transformer_encoder(RoPE_out)#(batch,seq_len,hidden_zise)
        mRNA_vector = torch.unsqueeze(torch.mean(transf_out, 1), 0)#(batch,1,hidden_zise)
        mRNA_vector = self.outlinear(mRNA_vector).reshape(1,1024)
        
        return mRNA_vector


# In[8]:


class ProteinGNNEncoder(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=1024, latent_dim=1024, heads=4, dropout=0.2):
        super(ProteinGNNEncoder, self).__init__()
        self.dropout = dropout

        # GATConv：(edge_dim=3)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=3, dropout=dropout)
        #  GATConv： hidden_channels * heads
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=3, dropout=dropout)

        # LayerNorm
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels)

        # latent_dim (1024)
        self.fc_encoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim)
        )
    def forward(self, x, edge_index, edge_attr):
        """
        in:
            data:torch_geometric.data.Data edge_index edge_attr
        out:
            latent: [batch_size, latent_dim]
        """
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)
        # GAT
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # GAT
        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm2(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)  # shape: [batch_size, hidden_channels*2]

        # [batch_size, latent_dim]
        latent = self.fc_encoder(x_graph)
        return latent


# In[9]:


#molgpt submodule
class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop) #dropout。
        self.resid_drop = nn.Dropout(config.resid_pdrop) #dropout。
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        #num = int(bool(config.num_props)) + int(config.scaffold_maxlen)   #int(config.lstm_layers)    #  int(config.scaffold) 
        #config.num_props：
        num = 1 #latent vector
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size() # B, T, C = batch size,input_len,embedding dimension

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save


# In[10]:


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn


# In[11]:


class molGPT(nn.Module):
    def __init__(self, config, stoi):
        super().__init__()
        self.config = config
        self.stoi = stoi
        self.vocab_size = len(stoi)
        self.tok_emb = nn.Embedding(self.vocab_size, config.n_embd)
        #self.type_emb = nn.Embedding(2, config.n_embd)
        self.latent_proj = nn.Linear(config.latent_dim, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.vocab_size, bias=False)
        self.block_size = config.block_size
        
    def forward(self, idx, targets=None, latent=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        if latent is None:
            latent = torch.zeros(t,config.latent_dim)
        latent_emb = self.latent_proj(latent).reshape(1,1,-1)
        #print("latent_emb: ",latent_emb)

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        #type_embeddings = self.type_emb(torch.ones((b, t), dtype=torch.long, device=idx.device))
        x = self.drop(token_embeddings + position_embeddings)# + type_embeddings
        #print(f"latent_emb.size():{latent_emb.size()}  x.size():{x.size()}")
        x = torch.cat([latent_emb, x], dim=1)
        #print(f"x = torch.cat([latent_emb, x], dim=1):{x}")

        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
        #print(f"x = self.ln_f(x) {x}")
        logits = self.head(x)

        logits = logits[:, 1:, :]

        loss = None
        #print("logits.size():", logits.size())
        #print("targets.size():", targets.size())
        #print(f"targets：{targets}")
        if targets is not None:
            #print(f"logits: {logits}")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss, attn_maps


# In[12]:


class Targetfeaturefusion(nn.Module):
    def __init__(self, ninp = 1024, nhead = 8, nhid= 1024, nlayers= 1, dropout=0.2):
        super(Targetfeaturefusion, self).__init__()
        self.kan1 = KANLayer(in_dim=ninp, out_dim=nhid)
        self.kan2 = KANLayer(in_dim=ninp, out_dim=nhid)
        self.kan3 = KANLayer(in_dim=ninp, out_dim=nhid)
        
    def forward(self,gene_fv, protstru_fv, protemb_fv):
        protemb_fv = protemb_fv.reshape(-1,1024)
        # print(f'gene_fv size: {gene_fv.size()} ')
        # print(f'protstru_fv size: {protstru_fv.size()} ')
        # print(f'protemb_fv size: {protemb_fv.size()} ')
        genekanout = self.kan1(gene_fv)[0]
        protskanout = self.kan2(protstru_fv)[0]
        protekanout = self.kan3(protemb_fv)[0]
        TargetfusionFeature, att_scores = attention(genekanout, protskanout, protekanout, mask=None, dropout=None,use_RoPE = False)
        #print("FusionFeature size: ",FusionFeature.size())
        FusionFeature = TargetfusionFeature.reshape(-1,1,1024)
        
        return FusionFeature


# In[13]:


class Multitargetfeaturefusion(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.kan1 = KANLayer(in_dim=input_dim*2, out_dim=hidden_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)           
    
    def forward(self, x1, x2):
        # [1, 1024] -> [1, 2048]
        x1 = x1.reshape(-1,1024)
        x2 = x2.reshape(-1,1024)
        x = torch.cat([x1, x2], dim=-1)
        x = self.kan1(x)[0]          # [1, 2048] -> [1, 2048]
        x = self.dropout(x)
        x, att_scores = attention(x, x, x, mask=None, dropout=None,use_RoPE = False)
        x = self.norm(x + (x1 + x2)/2)
        x = x.reshape(-1,1,1024)
        return x


# In[14]:


class MMDD (nn.Module):
    def __init__(self,molGPTconfig,smiles_stoi):
        super(MMDD, self).__init__()
        self.gencoder = mRNAEncoder()
        self.protencoder = ProteinGNNEncoder()
        self.targetfeaturefusion = Targetfeaturefusion()
        
        self.Multitargetfeaturefusion = Multitargetfeaturefusion()
        
        self.drugdecoder = molGPT(molGPTconfig, stoi = stoi)
        
    def forward(self, DrugG_in, DrugG_outpt, T1gene, T1protstru, T1protemb_fv, T2gene, T2protstru, T2protemb_fv,):
        
        T1gene_fv = self.gencoder(T1gene)
        #print(f"T1gene_fv:{T1gene_fv}")
        T2gene_fv = self.gencoder(T2gene)
        #print(f"T2gene_fv:{T2gene_fv}")
        T1protstru_fv = self.protencoder(T1protstru.x, T1protstru.edge_index, T1protstru.edge_attr)#, T1protstru_b
        #print(f"T1protstru_fv:{T1protstru_fv}")
        T2protstru_fv = self.protencoder(T2protstru.x, T2protstru.edge_index, T2protstru.edge_attr)#, T2protstru_b
        #print(f"T2protstru_fv:{T2protstru_fv}")
        T1Multi_fv = self.targetfeaturefusion(T1gene_fv,T1protstru_fv,T1protemb_fv)
        #print(f"T1Multi_fv:{T1Multi_fv}")
        T2Multi_fv = self.targetfeaturefusion(T2gene_fv,T2protstru_fv,T2protemb_fv)
        #print(f"T2Multi_fv:{T2Multi_fv}")
        Multi_T_Fv = self.Multitargetfeaturefusion(T1Multi_fv,T2Multi_fv)
        #print(f"Multi_T_Fv:{Multi_T_Fv}")
        logits, loss, _ = self.drugdecoder(DrugG_in, targets = DrugG_outpt, latent = Multi_T_Fv)#损失函数和优化算法整合出去
        
        return logits, loss, _
        
    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (nn.Linear, nn.LSTM)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = f"{mn}.{pn}" if mn else pn

                if "bias" in pn:
                    no_decay.add(fpn)
                elif "weight" in pn and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif "ln" in pn or "norm" in pn:
                    no_decay.add(fpn)
                elif "emb" in pn:
                    no_decay.add(fpn)
                else:
                    decay.add(fpn)

        if "pos_emb" in param_dict:
            no_decay.add("pos_emb")

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"{inter_params} in decay&no_decay set！"
        unclassified = param_dict.keys() - union_params
        if unclassified:
            print("unclassified")
            no_decay.update(unclassified)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)

        print("Decay parameters:", sorted(decay))
        print("No decay parameters:", sorted(no_decay))
        for i, group in enumerate(optim_groups):
            print(f"Optimizer group {i}:")
            print(f"  Parameters: {len(group['params'])}")
            print(f"  Weight decay: {group['weight_decay']}")

        return optimizer
 


# In[15]:

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

stoi = vocab['stoi']
itos = vocab['itos']

print("Vocabulary size:", len(stoi))
print("stoi example:", stoi)
print("itos example:", itos)

# In[16]:

class GPTConfig:
    def __init__(self, vocab_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = 128
        self.latent_dim = 1024
        self.n_layer = 8
        self.n_head = 8 
        self.n_embd = 256
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        for k, v in kwargs.items():
            setattr(self, k, v)


# In[17]:


config1 = GPTConfig(len(stoi))

MMDDmodelv5 = MMDD(config1,stoi)
print(f"Model size: {get_model_size(MMDDmodelv5):.2f} MB")
print("Submodule：")
for name, submodule in MMDDmodelv5.named_children():
    num_params = sum(p.numel() for p in submodule.parameters())
    print(f"{name}: {num_params} parameters")


# In[18]:


class MultiModalDualTargetDataset(Dataset):
    def __init__(self, csv_file, smiles_stoi, smiles_block_size, prot_emb_dir, prot_struct_dir):
        self.data = pd.read_csv(csv_file)
        self.stoi = smiles_stoi
        self.pad_token = smiles_stoi['<pad>']
        self.block_size = smiles_block_size
        self.prot_emb_dir = prot_emb_dir
        self.prot_struct_dir = prot_struct_dir

    def __len__(self):
        return len(self.data)

    def tokenize_smiles(self, smiles):
        tokens = re.findall(r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])", smiles)
        tokens = ['<start>'] + tokens + ['<end>']
        tokens = tokens[:self.block_size]
        x = [self.stoi.get(token, self.stoi.get('<unk>')) for token in tokens]
        if len(x) < self.block_size:
            x = x + [self.pad_token] * (self.block_size - len(x))
        return torch.tensor(x, dtype=torch.long)

    def tokenize_gene_seq(self, gene_seq_str):
        tokens = ast.literal_eval(gene_seq_str)
        return torch.tensor(tokens, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        smiles = row['SMILES']
        smiles_tensor = self.tokenize_smiles(smiles)

        gene_seq1 = self.tokenize_gene_seq(row['Target1_Encoded_Gene_Seq'])
        gene_seq2 = self.tokenize_gene_seq(row['Target2_Encoded_Gene_Seq'])

        target1 = row['Target1_Uniprot']
        target2 = row['Target2_Uniprot']

        emb1_path = os.path.join(self.prot_emb_dir, f"{target1}.npy")
        emb2_path = os.path.join(self.prot_emb_dir, f"{target2}.npy")
        struct1_path = os.path.join(self.prot_struct_dir, f"alphafold_{target1}.pt")
        struct2_path = os.path.join(self.prot_struct_dir, f"alphafold_{target2}.pt")

        prot_emb1 = torch.tensor(np.load(emb1_path), dtype=torch.float32)
        prot_emb2 = torch.tensor(np.load(emb2_path), dtype=torch.float32)
        prot_struct1 = torch.load(struct1_path)
        prot_struct2 = torch.load(struct2_path)

        return {
            'smiles': smiles_tensor,       # Tensor: [smiles_block_size]
            'gene_seq1': gene_seq1,         # Tensor: [L1]
            'prot_struct1': prot_struct1,   # torch_geometric.data.Data
            'prot_emb1': prot_emb1,         # Tensor: [E]
            'gene_seq2': gene_seq2,         # Tensor: [L2]
            'prot_struct2': prot_struct2,   # torch_geometric.data.Data
            'prot_emb2': prot_emb2          # Tensor: [E]
        }

def custom_collate(batch):

    batch_collated = {}
    batch_collated['smiles'] = torch.stack([item['smiles'] for item in batch], dim=0)
    batch_collated['gene_seq1'] = [item['gene_seq1'] for item in batch]
    batch_collated['gene_seq2'] = [item['gene_seq2'] for item in batch]
    batch_collated['prot_emb1'] = torch.stack([item['prot_emb1'] for item in batch], dim=0)
    batch_collated['prot_emb2'] = torch.stack([item['prot_emb2'] for item in batch], dim=0)
    batch_collated['prot_struct1'] = Batch.from_data_list([item['prot_struct1'] for item in batch])
    batch_collated['prot_struct2'] = Batch.from_data_list([item['prot_struct2'] for item in batch])
    return batch_collated

dataset = MultiModalDualTargetDataset(
    csv_file="../data/multidata/filtered_dual_target_data2.csv",
    smiles_stoi=stoi,
    smiles_block_size=128,
    prot_emb_dir="../data/multidata/prot_emb_data",
    prot_struct_dir="../data/multidata/prot_graph_data"
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)

for batch in dataloader:
    print(f"batch {batch}")
    print("SMILES tokens:", batch['smiles'])
    
    print("Target1:")
    print(f"gene size: {batch['gene_seq1'][0].size()} seq: {batch['gene_seq1']}")
    print("prot struct:", batch['prot_struct1'][0])
    print("prot emb:", batch['prot_emb1'].size(), batch['prot_emb1'])

    print("Target2:")
    print(f"gene size: {batch['gene_seq2'][0].size()} seq: {batch['gene_seq2']}")
    print("prot struct:", batch['prot_struct2'][0])
    print("prot emb:", batch['prot_emb2'].size(), batch['prot_emb2'])
    break

class TrainerConfig:
    def __init__(self, **kwargs):
        self.max_epochs = 20
        self.batch_size = 1
        self.latent_dim = 1024
        self.learning_rate = 2e-6
        self.betas = (0.9, 0.95)
        self.weight_decay = 0.1
        self.grad_norm_clip = 0.8
        self.lr_decay = False
        self.warmup_tokens = 375e6
        self.final_tokens = 260e9
        self.ckpt_path = None
        self.num_workers = 0

        for k, v in kwargs.items():
            setattr(self, k, v)


# In[20]:


train_config = TrainerConfig(
    max_epochs=20,
    batch_size=1,
    learning_rate=2e-6,
    weight_decay=0.1,
    grad_norm_clip=1.0,
    ckpt_path='MMDDv1.pt'
)

optimizer = MMDDmodelv5.configure_optimizers(train_config)

# In[21]:


def train(model, loader, optimizer, config, device='cuda'):
    model.train()
    scaler = GradScaler()
    tokens_processed = 0

    for epoch in range(0,50):
        total = len(loader)
        pbar = tqdm(enumerate(loader), total=len(loader))
        hours, minutes, seconds = time_elapsed(start_time)
        print(f" | Time elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        batchno = 0
        batchsave = 0
        for it, x in pbar:
            batchno += 1
            smiles_tensor = x['smiles'].to(device)
            target1_nuc = x['gene_seq1'][0].to(device)
            target1_graph = x['prot_struct1'][0].to(device)
            target1_emb = x['prot_emb1'].to(device)
            target2_nuc = x['gene_seq2'][0].to(device)
            target2_graph = x['prot_struct2'][0].to(device)
            target2_emb = x['prot_emb2'].to(device)
            inpt = smiles_tensor[:, :-1]
            outpt = smiles_tensor[:, 1:]
            with autocast():
                logits, loss, _ = model(inpt,outpt,target1_nuc,target1_graph,target1_emb,target2_nuc,target2_graph,target2_emb)
                loss = loss.mean()

            optimizer.zero_grad()

            if torch.isnan(loss) or torch.isinf(loss):
                print("❌ Loss is NaN or Inf, skipping step.")
                continue

            try:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
    
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                print(f"⚠️ Runtime error during scaler step: {e}")
                continue

            
            if config.lr_decay:
                tokens_processed += (outpt >= 0).sum()
                if tokens_processed < config.warmup_tokens:
                    lr_mult = float(tokens_processed) / float(max(1, config.warmup_tokens))
                else:
                    progress = float(tokens_processed - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config.learning_rate
            
            pbar.set_description(f"Epoch {epoch + 1}, Iter {it}/{len(loader)}: Loss {loss.item():.5f}, LR {lr:.6f}")
            if batchno == int(len(loader)/2):
                batchno = 0
                torch.save(model.state_dict(), f'MMDDmodelv5_epoch_{epoch+1}_batch_{batchsave+1}.pth')
                batchsave += 1

        torch.save(model.state_dict(), f'MMDDmodelv5_epoch_{epoch+1}.pth')

# In[22]:

start_time = time.time()
checkpoint = torch.load("modelv9_epoch_8_batch_1.pth", map_location="cpu")

filtered_state_dict = {
    k: v for k, v in checkpoint.items()
    if not k.startswith("type_emb")
}

missing_keys, unexpected_keys = MMDDmodelv5.drugdecoder.load_state_dict(filtered_state_dict, strict=False)


if missing_keys:
    print(f"missing: {missing_keys}")
if unexpected_keys:
    print(f"missing: {unexpected_keys}")

MMDDmodelv5 = MMDDmodelv5.to(device)
train(MMDDmodelv5, dataloader, optimizer, train_config, device=device)


# In[ ]:




