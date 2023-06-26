#!/usr/bin/env python
# coding: utf-8

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
from propose import models, Accuracy


# Set up GPU device
device = torch.device('cuda', 0) #'cpu' #torch.device('cuda', 5)

# Number of genes to select
num_genes = [128, 64, 32, 16, 12] #12, 16, 32, 64, 128]

ddir = '/bigstore/GeneralStorage/fangming/projects/visctx/data_dump/counts/'
f = os.path.join(ddir, "P38_1a2a_glut.h5ad")

study = "L234_allgenes_sep14"
resdir = '/bigstore/GeneralStorage/fangming/projects/visctx/data_dump/test_propose'
output_res = os.path.join(resdir, f'res_{study}.pkl')
output_eval = os.path.join(resdir, f'eval_{study}.pkl')
output_fig = os.path.join(resdir, f"fig_{study}.pdf")

adata = anndata.read(f)

raw = np.asarray(adata.X.todense())
cpm = (raw/raw.sum(axis=1).reshape(-1,1))*1e6
gene_names = adata.var.index.values
cell_types = adata.obs['Type']
cell_types_num = pd.Categorical(cell_types).codes

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


# # Run PROPOSE for cell type classification
# Set up datasets
train_dataset = ExpressionDataset(binary[train_inds], cell_types_num[train_inds])
val_dataset = ExpressionDataset(binary[val_inds], cell_types_num[val_inds])
test_dataset = ExpressionDataset(binary[test_inds], cell_types_num[test_inds])

propose_class_results = {}

# Set up selector
selector = PROPOSE(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   device=device,
                   hidden=[128, 128])

# Eliminate many candidates
candidates, model = selector.eliminate(target=500, mbsize=128, max_nepochs=600)

for num in num_genes:
    # Select specific number of genes
    inds, model = selector.select(num_genes=num, mbsize=128, max_nepochs=600)
    propose_class_results[num] = inds

# Save results
with open(output_res, 'wb') as f:
    pickle.dump(propose_class_results, f)


# # Prepare gene sets
# with open(output_res, 'rb') as f:
#     propose_class_results = pickle.load(f)
num_genes = list(propose_class_results.keys())

# Dictionary of methods
methods = {
    'PROPOSE-Class': propose_class_results,
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
            loss_fn=torch.nn.CrossEntropyLoss(),
            verbose=False)

        # Validation performance
        test_acc = model.validate(DataLoader(test_dataset, batch_size=1024), Accuracy()).item()
        results[name][num] = test_acc

# Save results
with open(output_eval, 'wb') as f:
    pickle.dump(results, f)


# # Plot results
# with open(output_eval, 'rb') as f:
#     results = pickle.load(f)
num_features = list(results['PROPOSE-Class'].keys())

# Make plot
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(num_features, 
        [results['PROPOSE-Class'][num] for num in num_features],
        marker='o', markersize=7, color='C2', label='PROPOSE-Class')
ax.legend(loc='lower right', fontsize=18, frameon=False)
ax.set_title('Cell Type Classification', fontsize=18)
ax.set_xlabel('# Genes', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
fig.tight_layout()
fig.savefig(output_fig, bbox_inches='tight', dpi=300)
plt.show()
