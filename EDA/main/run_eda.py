import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
##################################################
# %%
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holoviews as hv
import hvplot.pandas
import cv2
hv.extension('bokeh')

# parser = argparse.ArgumentParser()
# dataframe path
# parser.add_argument('--df_path', 
#                     default='data\Consumer2Shop\Consumer2Shop_box.csv',
#                     help='Dataframe contains the deep fashion dataset')

# %%
# args = parser.parse_args()
# Read csv
# clothes = {1: 'upper body', 2: 'lower body', 3: 'full body'}
# source = {1: 'Store', 2: 'Consumer'}
# df = pd.read_csv('D:\\HomeWork\\anaconda\\data\\Consumer2Shop\\Consumer2Shop_box.csv')
# df['clothes_type'] = df['clothes_type'].apply(lambda x: clothes.get(x))
# df['source_type'] = df['source_type'].apply(lambda x: source.get(x))

# # Number of image per class
# # %%
# df['label'] = df['image_name'].apply(lambda x: x.split('/')[2].replace('_', ' '))
# class_name = df['label'].unique()
# print(len(class_name))
# dcounts = df.label.value_counts(normalize=True)
# dcounts_df = pd.DataFrame({'class_name': dcounts.index.tolist(), 'percentage_image': dcounts})
# dcounts_df.reset_index(drop=True, inplace=True)
# plot = dcounts_df[0:17].hvplot.bar(x='class_name', y='percentage_image', invert=False, flip_yaxis=False, 
#                             height=600, width=800, ylim=(0,0.45))
# renderer = hv.renderer('bokeh')
# renderer.save(plot, 'test_xarray') 

        
# # %%
# table = df.groupby(['source_type', 'clothes_type']).size()
# print(table.head())
# plot2 = table.hvplot.bar(stacked=False, height=500, rot=60, title='Number of image per class')
# renderer = hv.renderer('bokeh')
# renderer.save(plot2, 'source_number_image') 
# # %%
# df1 = pd.read_csv('D:\\HomeWork\\anaconda\\data\\Shopping100k\\Attributes\\shopping100k.csv')
# df1['label'] = df1['image_name'].apply(lambda x: ' '.join(map(str, x.split('/')[1].split('_')[1:])))

# dcounts = df1.label.value_counts(normalize=False)
# dcounts_df = pd.DataFrame({'class_name': dcounts.index.tolist(), 'number_of_image': dcounts})
# dcounts_df.reset_index(drop=True, inplace=True)
# plot = dcounts_df[0:17].hvplot.bar(x='class_name', y='number_of_image', invert=True, flip_yaxis=True, 
#                             height=600, width=800, ylim=(0,30000))
# renderer = hv.renderer('bokeh')
# renderer.save(plot, 'shopping100k_number_of_image') 
# %%
df2 = pd.read_csv('D:\\HomeWork\\anaconda\\data\\Consumer2Shop\\siamese_data.csv')
def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def read_row(row):
    p1 = read_image('D:/HomeWork/anaconda/data/Consumer2Shop/'+row[0])
    p2 = read_image('D:/HomeWork/anaconda/data/Consumer2Shop/'+row[5])
    p3 = read_image('D:/HomeWork/anaconda/data/Consumer2Shop/'+row[10])

    return (p1, p2, p3)


# %%
randnum = np.random.randint(0, len(df2), size=9)
rand_row = df2.iloc[randnum, :].values.tolist()
for i in range(3):
    p1, p2, p3 = read_row(rand_row[i])
    plt.subplot(3,3, 1+i*3)
    plt.imshow(p1)
    plt.title('Anchor image')
    plt.axis('off')
    plt.subplot(3,3, 2+i*3)
    plt.imshow(p2)
    plt.title('Positive image')
    plt.axis('off')
    plt.subplot(3,3, 3+i*3)
    plt.imshow(p3)
    plt.title('Negative image')
    plt.axis('off')
plt.tight_layout()
plt.show()

# %%
precision_at_k = pd.read_csv('D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\metric\\precision_at_K.csv')
plot = precision_at_k.hvplot(x='k', y=['resnet50_1000 (0.682)', 'resnet50_2000 (0.687)', 'densenet161_128 (0.668)'],
             value_label='Top-k retrieval accuracy', height=600, width=1000,fontsize={'title': '20pt', 'ylabel': '17px', 'ticks': 10})
renderer = hv.renderer('bokeh')
renderer.save(plot, 'top_k_precision') 
# %%
reciprocal = pd.read_csv('D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\metric\\mean_reciprocal_rank.csv')
plot = reciprocal.hvplot(x='k', y=['resnet50_1000 (0.252)', 'resnet50_2000 (0.253)', 'densenet161_128 (0.248)'],
        value_label='Mean reciprocal rank', height=500, width=1000,fontsize={'title': '20pt','xlabel':'15px', 'ylabel': '17px', 'ticks': 10})
renderer = hv.renderer('bokeh')
renderer.save(plot, 'mean_reciprocal_rank') 

# %%
