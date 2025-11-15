#!/usr/bin/env python
# coding: utf-8

import pickle
import math
import logging
import re
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict

# Time calculation
def time_elapsed(start_time):
    elapsed = time.time() - start_time
    hours = int(elapsed / 3600)
    minutes = int((elapsed / 60) % 60)
    seconds = int(elapsed % 60)
    return hours, minutes, seconds

# Build vocabulary from SMILES file
def build_vocab(smiles_file):
    token_counts = defaultdict(int)
    with open(smiles_file, 'r') as f:
        for line in f:
            smiles = line.strip()
            tokens = re.findall(
                r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])",
                smiles)
            for token in tokens:
                token_counts[token] += 1

    stoi = {token: i for i, token in enumerate(token_counts.keys())}
    itos = {i: token for token, i in stoi.items()}

    # Add special tokens
    stoi['<pad>'] = len(stoi)
    stoi['<start>'] = len(stoi)
    stoi['<end>'] = len(stoi)
    itos[len(itos)] = '<pad>'
    itos[len(itos)] = '<start>'
    itos[len(itos)] = '<end>'

    return stoi, itos

# Build vocabulary and save
stoi, itos = build_vocab('../data/smiles.txt')
with open('vocab.pkl', 'wb') as f:
    pickle.dump({'stoi': stoi, 'itos': itos}, f)
print("vocab len:", len(stoi))
print("stoi:", stoi)
print("itos:", itos)

# SMILES Dataset
class SMILESDataset(Dataset):
    def __init__(self, smiles_file, stoi, block_size):
        self.smiles = open(smiles_file, 'r').read().splitlines()
        self.stoi = stoi
        self.pad_token = stoi['<pad>']
        self.block_size = block_size  # max sequence length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        tokens = re.findall(
            r"(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])",
            smiles)
        tokens = ['<start>'] + tokens + ['<end>']
        tokens = tokens[:self.block_size]
        x = [self.stoi[token] for token in tokens]
        if len(x) < self.block_size:
            x += [self.pad_token] * (self.block_size - len(x))
        return torch.tensor(x, dtype=torch.long)

# Dataloader
dataset = SMILESDataset('../data/pretrainedsmilesv6.txt', stoi, block_size=128)
loader = DataLoader(dataset, batch_size=256, shuffle=True)

# Causal Self-Attention Layer
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        num = 1  # latent vector
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num)).view(
            1, 1, config.block_size + num, config.block_size + num))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), attn_save

# Transformer Block
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

# GPT Model for Molecules
class molGPT(nn.Module):
    def __init__(self, config, stoi):
        super().__init__()
        self.config = config
        self.stoi = stoi
        self.vocab_size = len(stoi)
        self.tok_emb = nn.Embedding(self.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        self.latent_proj = nn.Linear(config.latent_dim, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, self.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias') or 'bias' in pn:
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or 'weight' in pn) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add('pos_emb')
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} in both decay and no_decay!"
        assert len(param_dict.keys() - union_params) == 0, f"Parameters {param_dict.keys() - union_params} not classified!"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, latent=None):
        b, t = idx.size()
        assert t <= self.block_size, "Input exceeds block size limit"
        if latent is None:
            latent = torch.zeros(t, self.config.latent_dim).to(idx.device)
        latent_emb = self.latent_proj(latent).unsqueeze(1)
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
        logits = logits[:, 1:, :]  # remove latent token

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, attn_maps

# Model Config
class GPTConfig:
    def __init__(self, vocab_size, block_size, latent_dim, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.latent_dim = latent_dim
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        for k, v in kwargs.items():
            setattr(self, k, v)

# Trainer Config
class TrainerConfig:
    def __init__(self, **kwargs):
        self.max_epochs = 10
        self.batch_size = 256
        self.latent_dim = 1024
        self.learning_rate = 1e-5
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

# Create and train model
config = GPTConfig(vocab_size=len(stoi), block_size=128, latent_dim=1024, n_layer=8, n_head=8, n_embd=256)
modelv8 = molGPT(config, stoi).to('cuda')
print("number of modelv8 parameters: %e", sum(p.numel() for p in modelv8.parameters()))

train_config = TrainerConfig(
    max_epochs=2,
    batch_size=128,
    learning_rate=1e-5,
    weight_decay=0.1,
    grad_norm_clip=1.0,
    ckpt_path='modelv8.pt'
)

optimizer = modelv8.configure_optimizers(train_config)

def train(model, loader, optimizer, config, device='cuda'):
    model.train()
    scaler = GradScaler()
    tokens_processed = 0

    for epoch in range(config.max_epochs):
        total = len(loader)
        pbar = tqdm(enumerate(loader), total=len(loader))
        hours, minutes, seconds = time_elapsed(start_time)
        print(f" | Time elapsed: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        batchno = 0
        batchsave = 0
        for it, x in pbar:
            batchno += 1
            x = x.to(device)
            latent = torch.zeros(x.size(0), config.latent_dim).to(device)
            inpt = x[:, :-1]
            outpt = x[:, 1:]

            with autocast():
                logits, loss, _ = model(inpt, outpt, latent=latent)
                loss = loss.mean()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            if config.lr_decay:
                tokens_processed += (outpt >= 0).sum()
                if tokens_processed < config.warmup_tokens:
                    lr_mult = float(tokens_processed) / float(max(1, config.warmup_tokens))
                else:
                    progress = float(tokens_processed - config.warmup_tokens) / float(
                        max(1, config.final_tokens - config.warmup_tokens))
                    lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                lr = config.learning_rate * lr_mult
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = config.learning_rate

            pbar.set_description(f"Epoch {epoch + 1}, Iter {it}/{len(loader)}: Loss {loss.item():.5f}, LR {lr:.6f}")
            if batchno == int(len(loader) / 10):
                batchno = 0
                batchsave += 1
                torch.save(model.state_dict(), f'modelv8_epoch_{epoch+1}_batch_{batchsave+1}.pth')

        torch.save(model.state_dict(), f'modelv8_epoch_{epoch+1}.pth')

start_time = time.time()
train(modelv8, loader, optimizer, train_config, device='cuda')
