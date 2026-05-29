import os
import copy
import baselines
import torch
import numpy as np
import pandas as pd
import functions as fn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import EGST

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)

import learner
from functions import CreateTimeIndexedDataset

use_cuda = True
device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
fn.set_seed(seed=2023, flag=True)
model_name = 'egst'
seq_l = 12  
pre_l = 6   
bs = 512    
p_epoch = 0
n_epoch = 300
law_list = np.array([-1.48, -0.74])
is_train = True
mode = 'completed'  
is_pre_train = False
patience = 10             
best_val_loss = float('inf')
epochs_no_improve = 0      
occ, prc, adj, col, dis, cap, time, inf = fn.read_dataset()
adj_dense = torch.Tensor(adj)
adj_sparse = adj_dense.to_sparse_coo().to(device)

train_occ, valid_occ, test_occ = fn.division(occ, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
train_prc, valid_prc, test_prc = fn.division(prc, train_rate=0.6, valid_rate=0.2, test_rate=0.2)
T = len(time)
n_train = int(T * 0.6)
n_valid = int(T * 0.8)

train_time = time.iloc[:n_train].reset_index(drop=True)
valid_time = time.iloc[n_train:n_valid].reset_index(drop=True)
test_time  = time.iloc[n_valid:].reset_index(drop=True)
train_dataset = fn.CreateDataset(train_occ, train_prc, seq_l, pre_l, device, adj_dense)
valid_dataset = fn.CreateDataset(valid_occ, valid_prc, seq_l, pre_l, device, adj_dense)
test_dataset  = fn.CreateDataset(test_occ,  test_prc,  seq_l, pre_l, device, adj_dense)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset), shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=len(test_dataset),  shuffle=False)
config = {
    'kcnn':2,
    'k':6,
    'm':2,
    'time_intervals': 300,
    'spatial_emb_dim': 6,
    'temp_dim_tid': 6,
    'if_spatial': True,
    'if_time_in_day': True,
    'if_time':True,
    'if_day_in_week': False,
    'device': device,
    'dynamic_emb_dim':8,
    'drop_edge_dyn': 0.10 
}
model = EGST(a_sparse=adj_sparse, config=config).to(device)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5,lr=1e-3)
loss_fn = torch.nn.MSELoss()
valid_loss = float('inf')
os.makedirs('./checkpoints', exist_ok=True)

if is_train:
    model.train()

    if is_pre_train:
        if mode == 'simplified':
            model = learner.fast_learning(
                law_list, model, model_name, p_epoch, bs,
                train_occ, train_prc, seq_l, pre_l, device, adj_dense
            )
        elif mode == 'completed':
            model = learner.physics_informed_meta_learning(
                law_list, model, model_name, p_epoch, bs,
                train_occ, train_prc, seq_l, pre_l, device, adj_dense
            )
        else:
            print("Mode error, skipping pre-training.")

    for epoch in tqdm(range(n_epoch), desc='Fine-tuning'):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            if len(batch) == 5:
                occupancy, price, label, time_idx, day_idx = batch
            else:
                occupancy, price, label = batch
                time_idx = None
                day_idx = None

            optimizer.zero_grad()
            pred = model(occupancy, price, time_idx, day_idx)

            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train = total_train_loss / len(train_loader)
        print(f"[Epoch {epoch}] Train Loss: {avg_train:.6f}")
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) == 5:
                    occupancy, price, label, time_idx, day_idx = batch
                else:
                    occupancy, price, label = batch
                    time_idx = None
                    day_idx = None
                pred = model(occupancy, price, time_idx, day_idx)
                loss = loss_fn(pred, label)
                total_val_loss += loss.item()
                print(f"Epoch {epoch} Validation Loss: {loss.item():.6f}")
        avg_val = total_val_loss / len(valid_loader)
        print(f"[Epoch {epoch}] Validation Loss: {avg_val:.6f}")
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            ckpt_path = f'./checkpoints/{model_name}_{pre_l}_bs{bs}_{mode}.pt'
            torch.save(model, ckpt_path)
            print(f"  ↓ Validation improved, saving to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  ↑ No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
            break

ckpt_path = f'./checkpoints/{model_name}_{pre_l}_bs{bs}_{mode}.pt'
model = torch.load(ckpt_path)
model.to(device)
model.eval()
predict_list = np.zeros((1, adj_dense.shape[1]))
label_list   = np.zeros((1, adj_dense.shape[1]))
for batch in test_loader:
    if len(batch) == 5:
        occupancy, price, label, time_idx, day_idx = batch
    else:
        occupancy, price, label = batch
        time_idx = None
        day_idx = None

    print('occupancy:', occupancy.shape, 'price:', price.shape, 'label:', label.shape)
    with torch.no_grad():
        pred = model(occupancy, price, time_idx, day_idx)

        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        predict_list = np.concatenate((predict_list, pred), axis=0)
        label_list   = np.concatenate((label_list, label), axis=0)
results = fn.metrics(test_pre=predict_list[1:], test_real=label_list[1:])
result_df = pd.DataFrame(
    columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE', 'R2'],
    data=[results]
)

os.makedirs('./results', exist_ok=True)
result_df.to_csv(f'./results/{model_name}_{pre_l}bs{bs}.csv', encoding='gbk', index=False)
