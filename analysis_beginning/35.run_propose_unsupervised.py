import numpy as np
import pandas as pd
import anndata
import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "/bigstore/GeneralStorage/fangming/projects/visctx/propose")
# from dredFISH.Utils import powerplots
from propose import PROPOSE, HurdleLoss, ExpressionDataset
from propose import models, Accuracy, MSELoss

# Set up GPU device (use torchgpu on blue)
# device = torch.device('cuda', 0) #'cpu' #torch.device('cuda', 5)
device = torch.device('cpu')

# Number of genes to select
num_genes = [200, 300, 400, 500, 600] 

ddir = '/bigstore/GeneralStorage/fangming/projects/visctx/data_dump/counts/'
f = os.path.join(ddir, "P38_1a2a_glut.h5ad")

study = "merfish_L234_allgenes_unsupervised_oct13"
resdir = '/bigstore/GeneralStorage/fangming/projects/visctx/data_dump/test_propose'
output_res = os.path.join(resdir, f'res_{study}.pkl')
output_eval = os.path.join(resdir, f'eval_{study}.pkl')
output_fig = os.path.join(resdir, f"fig_{study}.pdf")

adata = anndata.read(f)

raw = np.asarray(adata.X.todense())
cpm = (raw/raw.sum(axis=1).reshape(-1,1))*1e6
gene_names = adata.var.index.values

# Generate logarithmized and binarized data
binary = (raw > 0).astype(np.float32)
log = np.log(1 + raw)
logcpm = np.log(1 + cpm)

# For data splitting
n = len(raw)
n_train = int(0.8 * n)
n_test = int(0.1 * n)
all_rows = np.arange(n)
np.random.seed(0)
np.random.shuffle(all_rows)
train_inds = all_rows[:n_train]
val_inds = all_rows[n_train:-n_test]
test_inds = all_rows[-n_test:]
print(f'{n} total examples, {len(train_inds)} training examples, {len(val_inds)} validation examples, {len(test_inds)} test examples')

# Set up datasets
train_dataset = ExpressionDataset(binary[train_inds], logcpm[train_inds])
val_dataset = ExpressionDataset(binary[val_inds], logcpm[val_inds])
test_dataset = ExpressionDataset(binary[test_inds], logcpm[test_inds])

# Set up selector
propose_results = {}
selector = PROPOSE(train_dataset,
                   val_dataset,
                   loss_fn=HurdleLoss(),
                   device=device,
                   hidden=[128, 128])

# Eliminate many candidates
candidates, model = selector.eliminate(target=1000, mbsize=32, max_nepochs=100, tol=0.3)

for num in num_genes:
    # Select specific number of genes
    inds, model = selector.select(num_genes=num, mbsize=128, max_nepochs=500)
    ### hack to prevent duplicated genes (very rare but exists)
    inds = np.unique(inds)
    num = len(inds)
    propose_results[num] = inds

# Save results
with open(output_res, 'wb') as f:
    pickle.dump(propose_results, f)

# evaluate results
num_genes = list(propose_results.keys())
# Dictionary of methods
methods = {
    'PROPOSE': propose_results,
}
# Dictionary of results
results = {name: {} for name in methods}

# Fit models
for name in methods:
    for num in num_genes:
        # Get inds
        inds = methods[name][num]
        
        # Set up datasets
        train_dataset.set_inds(inds)
        val_dataset.set_inds(inds)
        test_dataset.set_inds(inds)
        
        # Train model
        model = models.MLP(
            input_size=num,
            output_size=train_dataset.output_size,
            hidden=[128, 128]).to(device)
        model.fit(
            train_dataset,
            val_dataset,
            mbsize=512,
            max_nepochs=500,
            loss_fn=MSELoss(),
            verbose=False)

        # Validation performance
        test_mse = model.validate(DataLoader(test_dataset, batch_size=1024), MSELoss()).item()
        results[name][num] = test_mse

# Save results
with open(output_eval, 'wb') as f:
    pickle.dump(results, f)