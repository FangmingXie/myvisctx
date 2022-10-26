#!/usr/bin/env python
# coding: utf-8

# # Load data

# In[1]:


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
from propose import PROPOSE, HurdleLoss, ExpressionDataset
from propose import models, Accuracy


# Set up GPU device
device = torch.device('cuda', 0) #'cpu' #torch.device('cuda', 5)
# In[2]:


ddir = '/bigstore/GeneralStorage/fangming/projects/visctx/data_dump/counts/'
resdir = '/bigstore/GeneralStorage/fangming/projects/visctx/data_dump/test_propose'
f = os.path.join(ddir, "P38_1a2a_glut.h5ad")
adata = anndata.read(f)
adata


# In[3]:


raw = np.asarray(adata.X.todense())
raw


# In[4]:


cpm = (raw/raw.sum(axis=1).reshape(-1,1))*1e6
cpm


# In[5]:


gene_names = adata.var.index.values

cell_types = adata.obs['Type']
cell_types_num = pd.Categorical(cell_types).codes
cell_types, cell_types_num


# In[6]:


# Generate logarithmized and binarized data
binary = (raw > 0).astype(np.float32)
log = np.log(1 + raw)
logcpm = np.log(1 + cpm)


# In[7]:


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

# In[8]:




# In[9]:




# In[10]:


# # Make cell types numerical
# cell_types = pd.Categorical(cell_types_98).codes

# Set up datasets
train_dataset = ExpressionDataset(binary[train_inds], cell_types_num[train_inds])
val_dataset = ExpressionDataset(binary[val_inds], cell_types_num[val_inds])


# In[11]:


# Number of genes to select
num_genes = [16]
propose_class_results = {}


# In[12]:


# Set up selector
selector = PROPOSE(train_dataset,
                   val_dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(),
                   device=device,
                   hidden=[128, 128])

# Eliminate many candidates
candidates, model = selector.eliminate(target=500, mbsize=128, max_nepochs=600)


# In[ ]:


for num in num_genes:
    # Select specific number of genes
    inds, model = selector.select(num_genes=num, mbsize=128, max_nepochs=600)
    propose_class_results[num] = inds


# In[ ]:


# Save results
with open(os.path.join(resdir, 'propose_class_results_v2.pkl'), 'wb') as f:
    pickle.dump(propose_class_results, f)


# # Cell type classification metric

# In[ ]:


# Set up datasets
train_dataset = ExpressionDataset(binary[train_inds], cell_types_num[train_inds])
val_dataset = ExpressionDataset(binary[val_inds], cell_types_num[val_inds])
test_dataset = ExpressionDataset(binary[test_inds], cell_types_num[test_inds])


# In[ ]:


# Prepare gene sets
with open(os.path.join(resdir, 'propose_class_results_v2.pkl'), 'rb') as f:
    propose_class_results = pickle.load(f)
num_genes = list(propose_class_results.keys())

# Dictionary of methods
methods = {
    'PROPOSE-Class': propose_class_results,
}

# Dictionary of results
results = {name: {} for name in methods}


# In[ ]:


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


# In[ ]:


# Save results
with open(os.path.join(resdir, 'classification_metric_v2.pkl'), 'wb') as f:
    pickle.dump(results, f)


# In[ ]:


# Plot results
with open(os.path.join(resdir, 'classification_metric_v2.pkl'), 'rb') as f:
    results = pickle.load(f)
num_features = list(results['PROPOSE-Class'].keys())


# In[ ]:


# Make plot
plt.figure(figsize=(9, 6))
plt.plot(num_features, [results['PROPOSE-Class'][num] for num in num_features],
         marker='o', markersize=7, color='C2', label='PROPOSE-Class')
plt.legend(loc='lower right', fontsize=18, frameon=False)
plt.tick_params(labelsize=14)
plt.title('Cell Type Classification', fontsize=18)
plt.xlabel('# Genes', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




