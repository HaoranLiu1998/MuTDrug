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


def extract_sequence(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    sequence = ''.join(line.strip() for line in lines if not line.startswith('>'))
    return sequence

def encode_sequence(seq):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    encoded_seq = []
    for char in seq.upper():
        if char not in mapping:
            return None
        encoded_seq.append(mapping[char])
    return encoded_seq

def tokenize_gene_seq(gene_seq_str):
    tokens = ast.literal_eval(gene_seq_str)
    return torch.tensor(tokens, dtype=torch.long)


# In[4]:


#Set device
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    device = torch.device("cuda")
    cuda = True
else:
    device = torch.device("cpu")
    cuda = False
    
print("Device =", device)


# In[5]:


# USE_CUDA = torch.cuda.is_available() and torch.cuda.device_count() > 1

# if USE_CUDA:
#     device = torch.device("cuda:1")
#     cuda = True
# else:
#     device = torch.device("cpu")
#     cuda = False

# print("Device =", device)


# In[6]:


#Set time
def time_elapsed(start_time):
    elapsed = time.time() - start_time  
    hours = int(elapsed/3600)           
    minutes = int(int(elapsed/60)%60)   
    seconds = int(elapsed%60)           
    
    return hours, minutes, seconds


# In[7]:


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


# In[8]:


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


# In[9]:


class ProteinGNNEncoder(nn.Module):
    def __init__(self, in_channels=20, hidden_channels=1024, latent_dim=1024, heads=4, dropout=0.2):
        super(ProteinGNNEncoder, self).__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, edge_dim=3, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=3, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)
        self.norm2 = nn.LayerNorm(hidden_channels)

        self.fc_encoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(latent_dim)
        )
    def forward(self, x, edge_index, edge_attr):

        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.norm2(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=1)  # shape: [batch_size, hidden_channels*2]

        latent = self.fc_encoder(x_graph)
        return latent


# In[10]:


#molgpt submodule
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
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

        y = self.resid_drop(self.proj(y))
        return y, attn_save


# In[11]:


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


# In[12]:


class molGPT(nn.Module):
    def __init__(self, config, stoi):
        super().__init__()
        self.config = config
        self.stoi = stoi
        self.vocab_size = len(stoi)
        self.tok_emb = nn.Embedding(self.vocab_size, config.n_embd)
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

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = torch.cat([latent_emb, x], dim=1)

        attn_maps = []
        for layer in self.blocks:
            x, attn = layer(x)
            attn_maps.append(attn)

        x = self.ln_f(x)
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


# In[13]:


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


# In[14]:


class Multitargetfeaturefusion(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.kan1 = KANLayer(in_dim=input_dim*2, out_dim=hidden_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)           
    
    def forward(self, x1, x2):
        x1 = x1.reshape(-1,1024)
        x2 = x2.reshape(-1,1024)
        x = torch.cat([x1, x2], dim=-1)
        x = self.kan1(x)[0]          # [1, 2048] -> [1, 2048]
        x = self.dropout(x)
        x, att_scores = attention(x, x, x, mask=None, dropout=None,use_RoPE = False)
        x = self.norm(x + (x1 + x2)/2)
        x = x.reshape(-1,1,1024)
        return x

# In[15]:


class MMDD (nn.Module):
    def __init__(self,molGPTconfig,smiles_stoi):
        super(MMDD, self).__init__()
        self.gencoder = mRNAEncoder()
        self.protencoder = ProteinGNNEncoder()
        self.targetfeaturefusion = Targetfeaturefusion()
        
        self.Multitargetfeaturefusion = Multitargetfeaturefusion()
        
        self.drugdecoder = molGPT(molGPTconfig, stoi = stoi)
        
    def forward(self, DrugG_in=0, DrugG_outpt=0, T1gene=0, T1protstru=0, T1protemb_fv=0,
                T2gene=0, T2protstru=0, T2protemb_fv=0,encode = False, latent= 0, generate = False):
        if encode == True:
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
            
            return Multi_T_Fv
        
        if generate == True:
            #print(f"Multi_T_Fv:{Multi_T_Fv}")
            logits, loss, _ = self.drugdecoder(DrugG_in, latent = latent)
            return logits
        
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
 


# In[16]:


with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
stoi = vocab['stoi']
itos = vocab['itos']


# In[17]:


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


# In[18]:


config1 = GPTConfig(len(stoi))

MMDDmodelv3 = MMDD(config1,stoi)
print(f"Model size: {get_model_size(MMDDmodelv3):.2f} MB")
print("Submodule：")
for name, submodule in MMDDmodelv3.named_children():
    num_params = sum(p.numel() for p in submodule.parameters())
    print(f"{name}: {num_params} parameters")


# In[19]:


targetfilename = "mental_disease_target"
prot_emb = np.load("C:/Users/LHR/0code/NC2025/result/target/"+targetfilename+"/emb/1TAAR1_2DRD2_emb.npy")
prot_emb_tensor = torch.from_numpy(prot_emb)
T1protemb_fv = prot_emb_tensor[0]
T2protemb_fv = prot_emb_tensor[1]

T1gene_file = "C:/Users/LHR/0code/NC2025/result/target/"+targetfilename+"/gene/1NM_138327.4.txt"
T1gene = tokenize_gene_seq(repr(encode_sequence(extract_sequence(T1gene_file))))
T2gene_file = "C:/Users/LHR/0code/NC2025/result/target/"+targetfilename+"/gene/2NM_000795.4.txt"
T2gene = tokenize_gene_seq(repr(encode_sequence(extract_sequence(T2gene_file))))


T1protstru = torch.load("C:/Users/LHR/0code/NC2025/result/target/"+targetfilename+"/stru/18w8a_pure.pt")
T2protstru = torch.load("C:/Users/LHR/0code/NC2025/result/target/"+targetfilename+"/stru/26cm4_pure.pt")


# In[20]:


print("T1protemb_fv: ",T1protemb_fv.shape)
print("T2protemb_fv: ",T2protemb_fv.shape)
print("T1gene: ",T1gene.shape)
print("T2gene: ",T2gene.shape)
print("T1protstru: ",T1protstru)
print("T2protstru: ",T2protstru)


# In[21]:


def generate_and_save_smiles(model, stoi, itos, output_file, temperature, num_samples=100, max_length=128, device='cuda', latent_input=None):
    model.eval()
    generated_smiles_list = []

    with torch.no_grad():
        for _ in range(num_samples):
            idx = torch.tensor([[stoi['<start>']]], dtype=torch.long, device=device)
            generated_tokens = []

            for _ in range(max_length):
                logits = model(idx, latent=latent_input,generate = True)
                logits = logits[:, -1, :] / temperature
                logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if next_token.item() == stoi['<end>']:
                    break
                generated_tokens.append(next_token.item())
                idx = torch.cat([idx, next_token], dim=1)

            generated_smiles = ''.join([itos[token] for token in generated_tokens])
            generated_smiles = generated_smiles.replace(itos[stoi['<pad>']], '').replace(itos[stoi['<start>']], '').replace(itos[stoi['<end>']], '')
            generated_smiles_list.append(generated_smiles)

    with open(output_file, 'w') as f:
        for smiles in generated_smiles_list:
            f.write(smiles + '\n')

    print(f"Generated {num_samples} SMILES and saved to {output_file}")


# In[22]:

config1 = GPTConfig(len(stoi))
MMDDmodelv3 = MMDD(config1,stoi)

for numep in range(50,51):
    for numbat in range(1,2):
        MMDDmodelv3.load_state_dict(torch.load('E:/AAAI2025/model/MMDDmodelv5_epoch_%d_batch_%d.pth'%(numep,numbat)))
        MMDDmodelv3.to(device)
        T1gene = T1gene.to(device)
        T1protstru = T1protstru.to(device)
        T1protemb_fv = T1protemb_fv.to(device)
        T2gene = T2gene.to(device)
        T2protstru = T2protstru.to(device)
        T2protemb_fv = T2protemb_fv.to(device)
        latent_input = MMDDmodelv3(T1gene=T1gene, T1protstru=T1protstru, T1protemb_fv=T1protemb_fv,
                                   T2gene=T2gene, T2protstru=T2protstru, T2protemb_fv=T2protemb_fv, encode = True)
        for tempera in range(10,11,1):
            tempera = tempera/10
            output_file_path ='../result/gensmiles/'+targetfilename+'_more/V5gensmiles_epoch_%d_batch_%d_tempera%.2f.txt'%(numep,numbat,tempera)
            print("output_file_path: ",output_file_path)
            
            generate_and_save_smiles(
                MMDDmodelv3,
                stoi,
                itos,
            
                output_file=output_file_path,
                temperature=tempera,
                num_samples=1000,
                max_length=128,
                device = device,
                latent_input = latent_input
    )


# In[ ]:





# In[ ]:




