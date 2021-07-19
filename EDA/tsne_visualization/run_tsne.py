import os
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import pandas as pd
import cv2

df = pd.read_csv('')

LOG_DIR = 'D:\\HomeWork\\anaconda\\fashion-visual-search\\src\\EDA\\tsne_visualization\\logs\\shopping100k\\'

def create_sprite_image(sample_file, df_path, img_dir, image_name):
    # Function to convert data to image
    def _images_to_sprite(data):
        """Creates the sprite image along with any necessary padding
        Args:
        data: NxHxW[x3] tensor containing the images.
        Returns:
        data: Properly shaped HxWx3 image with any necessary padding.
        """
        if len(data.shape) == 3:
            data = np.tile(data[...,np.newaxis], (1,1,1,3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) - min).transpose(3,0,1,2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1,2,3,0) / max).transpose(3,0,1,2)
        # Inverting the colors seems to look better for MNIST
        #data = 1 - data

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                constant_values=0)
        # Tile the individual thumbnails into an image.
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data
    
    # Dataframe is used create sprite image
    df = pd.read_csv(df_path)
    
    # sample file contain sample idx with its embedding
    sample = np.load(sample_file, allow_pickle=True)
    sample = list(sample.tolist())
    
    # image data to store data images
    img_data=[]
    for path in df.loc[sample, 'image_name'].values.tolist():
        input_img = cv2.imread(img_dir + '/'+ path)
        input_img_resized=cv2.resize(input_img,(224,224))
        img_data.append(input_img_resized)

    img_data = np.array(img_data)
    sprite = _images_to_sprite(img_data)
    print(img_data.shape[1])
    cv2.imwrite(os.path.join(LOG_DIR, image_name), sprite)

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    # Create sprite image
    create_sprite_image('fashion-visual-search\\src\\EDA\\tsne_visualization\\Sample_tsne.npy','data\Shopping100k\Attributes\shopping100k.csv', 'data/Shopping100k/Images', 'sprite_visual.png')

# Load embedded vectors
feature_vectors = np.loadtxt(os.path.join(LOG_DIR,'tsne_visual.tsv'), delimiter='\t')
feature_vectors = tf.Variable(feature_vectors, name='features')
print(feature_vectors)

checkpoint = tf.train.Checkpoint(embedding=feature_vectors)
checkpoint.save(os.path.join(LOG_DIR, 'embedding.ckpt'))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding.metadata_path =  'metadata.tsv'
embedding.sprite.image_path = 'sprite_visual.png'
embedding.sprite.single_image_dim.extend([224, 224])
# Saves a config file that TensorBoard will read during startup.
projector.visualize_embeddings(LOG_DIR, config)