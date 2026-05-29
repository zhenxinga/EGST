
import torch
import torch.nn as nn
import torch.nn.functional as F
import functions as fn
import os
import math



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)

class EGST(nn.Module):
    def __init__(self, a_sparse, config=None):
        super().__init__()
        default = {
            'seq': 12, 'kcnn': 2, 'k': 6, 'm': 2,
            'spatial_emb_dim': 4, 'temp_dim_tid': 8,
            'time_intervals': 300,    
            'if_spatial': False,
            'if_time_in_day': False,  
        }
        cfg = {**default, **(config or {})}
        self.seq = cfg['seq'] - cfg['kcnn'] + 1
        self.if_spatial = cfg['if_spatial']             
        self.if_time = cfg['if_time_in_day']            
        self.nodes = a_sparse.shape[0]                  
        self.dynamic_emb_dim = cfg['dynamic_emb_dim']    

        if self.if_spatial:
            d = cfg['spatial_emb_dim']
            self.node_emb = nn.Parameter(torch.empty(self.nodes, d, device=device))
            nn.init.xavier_uniform_(self.node_emb)

            self.node_proj = nn.Linear(d, self.seq, device=device)
            self.alpha_spatial = nn.Parameter(torch.tensor(0.01, device=device))  

        if self.if_time:
            slots = int(86400 / cfg['time_intervals']) 
            dt = cfg['temp_dim_tid']        
            self.time_emb = nn.Parameter(torch.empty(slots, dt, device=device)) 
            nn.init.xavier_uniform_(self.time_emb)  
            self.time_proj = nn.Linear(dt, self.seq, device=device)

        self.conv = nn.Conv2d(1, 1, (cfg['kcnn'], 2), device=device)

        adj = a_sparse.to_dense()   
        deg = adj.sum(1)
        Dm = torch.diag(torch.pow(deg + 1e-6, -0.5))
        adj_norm = Dm @ adj @ Dm    
        self.register_buffer('adj_norm', adj_norm)

        self.gcn1 = nn.Linear(self.seq, self.seq, device=device)
        self.gcn2 = nn.Linear(self.seq, self.seq, device=device)
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

        self.dynamic_node_emb = nn.Parameter(torch.empty(self.nodes, self.dynamic_emb_dim, device=device))
        nn.init.xavier_uniform_(self.dynamic_node_emb)
        self.gcn1_dyn = nn.Linear(self.seq, self.seq, device=device)
        self.gcn2_dyn = nn.Linear(self.seq, self.seq, device=device)

        self.trans_input_proj = nn.Linear(self.seq, 12, device=device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=12, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.predictor = nn.Sequential(
            nn.Linear(self.seq+1, (self.seq+1) // 2, device=device),
            nn.ReLU(),
            nn.Linear((self.seq +1)// 2, 1, device=device)
        )

        self.alpha1 = nn.Parameter(torch.tensor(0.5, device=device))
        self.alpha2 = nn.Parameter(torch.tensor(0.5, device=device))    
        self.alpha3 = nn.Parameter(torch.tensor(0.5, device=device))
        self.alpha4 = nn.Parameter(torch.tensor(0.5, device=device))  
        self.alpha5 = nn.Parameter(torch.tensor(0.1, device=device))  
   
    def build_dynamic_adj(self, occ, prc=None, topk=6, alpha=0.7, tau=0.1, mix_static=0,
                      detach=True, sym=True):
        B, N, S = occ.shape
        X = occ.detach() if detach else occ
        X = X - X.mean(dim=-1, keepdim=True)
        X = X / (X.std(dim=-1, keepdim=True) + 1e-6)
        S_occ = torch.matmul(X, X.transpose(1,2)) / S

        if prc is not None:
            S = S_occ

        if sym:
            S = 0.5 * (S + S.transpose(1,2))

        eye = torch.eye(N, device=S.device).unsqueeze(0)
        S = S.masked_fill(eye.bool(), float('-inf'))

        if topk is not None and topk < N:
            vals, idx = torch.topk(S, k=topk, dim=-1)
            S_topk = torch.full_like(S, float('-inf'))
            S_topk.scatter_(dim=-1, index=idx, src=vals)
            S = S_topk

        with torch.no_grad():
            A_stat = self.adj_norm
            A_stat_rw = A_stat / (A_stat.sum(dim=-1, keepdim=True) + 1e-6)

        A_dyn = (1 - mix_static) * A_dyn + mix_static * A_stat_rw
        
        return A_dyn


    def forward(self, occ, prc, time_idx=None, day_idx=None):
        B, N, S = occ.shape

        x = torch.stack([occ, prc], dim=3)
        x = x.view(B * N, S, 2).unsqueeze(1)
        h0 = self.conv(x).squeeze(-1).squeeze(1)
        h0 = h0.view(B, N, self.seq)

        aux = 0.
        if self.if_time and time_idx is not None:
            sp = self.node_proj(self.node_emb)          
            sp = sp.unsqueeze(0).repeat(B, 1, 1)         
        h = torch.cat([h0, sp], dim=-1)

        h1 = torch.einsum('ij,bjk->bik', self.adj_norm, h) #h1:[512,247,11]
        h1_flat = h1.reshape(B * N, self.seq) #[126464,11]
        h1_proj = self.gcn1(h1_flat)#[126464,11]
        h1_act = self.act(h1_proj)#[126464,11]
        h1 = h1_act.reshape(B, N, self.seq)#[512,247,11]
        h1 = self.dropout(h1)

        h2 = torch.einsum('ij,bjk->bik', self.adj_norm, h1) #[512,247,12]
        h2_flat = h2.reshape(B * N, self.seq)   #[126464,11]
        h2_proj = self.gcn2(h2_flat)    #[126464,11]
        h2_act = self.act(h2_proj) 
        h2 = h2_act.reshape(B, N, self.seq) #[512,247,11]
        h2 = self.dropout(h2)
        h1 = self.alpha1 * h1 + (1 - self.alpha1) * h   
        h2 = self.alpha2 * h2 + (1 - self.alpha2) * h1  
        adj_dyn = self.build_dynamic_adj(occ, prc=prc, topk=6, alpha=1, tau=0.2, mix_static=0)
        h1_dyn = torch.einsum('bij,bjk->bik', adj_dyn, h)
        h1_dyn_flat = h1_dyn.reshape(B * N, self.seq)
        h1_dyn_proj = self.gcn1_dyn(h1_dyn_flat)
        h1_dyn_act = self.act(h1_dyn_proj)
        h1_dyn = h1_dyn_act.reshape(B, N, self.seq)
        h1_dyn = self.dropout(h1_dyn)

        h2_dyn = torch.einsum('bij,bjk->bik', adj_dyn, h1_dyn)
        h2_dyn_flat = h2_dyn.reshape(B * N, self.seq)
        h2_dyn_proj = self.gcn2_dyn(h2_dyn_flat)
        h2_dyn_act = self.act(h2_dyn_proj)
        h2_dyn = h2_dyn_act.reshape(B, N, self.seq)
        h2_dyn = self.dropout(h2_dyn)
        f = (h1 + h2 +h1_dyn+h2_dyn)/4   # [B, N, seq]
        x_trans = f.view(B * N, self.seq)
        x_trans = self.trans_input_proj(x_trans).unsqueeze(1)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans[:, 0, :]
        y = self.predictor(x_trans).view(B, N)
        return y
