#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# %%
print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
X = np.load("D:\\HomeWork\\anaconda\\data\\model_inference\\shopping100k\\Multinet\\multinet_resnet50_2000\\ckpt4\\data_embeddings.npy")
df = pd.read_csv('D:\\HomeWork\\anaconda\\data\\Shopping100k\\Attributes\\shopping100k.csv')
df['label'] = df['image_name'].apply(lambda x: ' '.join(map(str, x.split('/')[1].split('_')[1:])))
sample = np.load('D:\\HomeWork\\anaconda\\data\\Shopping100k\\Sample_tsne.npy', allow_pickle=True)
sample = list(sample.tolist())
# %%


# %%
arr = X[sample]
pd.DataFrame(arr).to_csv(path_or_buf="D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\file2000.tsv",sep='\t',index=False, header=False)
# %%
pd.DataFrame(df['label'].values[sample]).to_csv(path_or_buf="D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\label.tsv",sep='\t',index=False, header=False)
# %%
file = open('D:\\HomeWork\\anaconda\\data\\Shopping100k\\shorter_sim_attr.txt')
lines = file.readlines()
num_list = [*range(len(lines))]
np.random.seed(2021)
sample_idx = np.random.choice(num_list, size=10, replace=False)
print(sample_idx)
choose = []
count = 0
for line in lines:
    select_list = line.split()
    select_list = [int(i) for i in select_list]
    if count in sample_idx:
        choose.extend(select_list)
    count+=1
np.save('D:\\HomeWork\\anaconda\\data\\Shopping100k\\Sample_tsne.npy', np.asarray(set(choose)))
# %%
