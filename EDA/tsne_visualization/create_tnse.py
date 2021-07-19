import numpy as np
import pandas as pd

# load embedding data for retrieving embedding
emb_data = np.load("D:\\HomeWork\\anaconda\\data\\model_inference\\shopping100k\\Multinet\\multinet_resnet50_2000\\ckpt4\\data_embeddings.npy")
# Load dataframe 
df = pd.read_csv('D:\\HomeWork\\anaconda\\data\\Shopping100k\\Attributes\\shopping100k.csv')
df['label'] = df['image_name'].apply(lambda x: ' '.join(map(str, x.split('/')[1].split('_')[1:])))
# 

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

sample = np.load('D:\\HomeWork\\anaconda\\data\\Shopping100k\\Sample_tsne.npy', allow_pickle=True)
sample = list(sample.tolist())

arr = emb_data[sample]
pd.DataFrame(arr).to_csv(path_or_buf="D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\tsne_sample_embed.tsv",sep='\t',index=False, header=False)
pd.DataFrame(df['label'].values[sample]).to_csv(path_or_buf="D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\label.tsv",sep='\t',index=False, header=False)
