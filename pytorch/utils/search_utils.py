import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


    
def pil_loader(path, bbox=None):
    """
    Function to read image and crop if have bounding box
    Args:
        path: Path to the image
        bbox: list bounding box consist of [xmin, ymin, xmax, ymax]
    
    Return:
        PIL image 
    """
    IMG = Image.open(path)
    if bbox:
        IMG = IMG.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    
    return IMG.convert('RGB')



def kNN_model(X, k):
    """
    Function to train an NearestNeighbors model, use to improve the speed of retrieving image from embedding database
    Args:
        X: data to train has shape MxN
        k: number of max nearest neighbors want to search
    
    Return:
        Nearest Neigbors model
    """
    nn_model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn_model.fit(X)
    return nn_model

def visualize(indexes, dataframe, labels,dir, cols=5):
    """
    Use to plot images
    Args:
        indexes: list of indexes to access dataframe
        dataframe: dataframe contains image path,...
        labels: label of image you want to plot
        dir: directory to the image
        cols: number of columns you want
    Return:
        None
    """
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.figsize'] = [25, 25]

    rows = len(indexes) // cols
    for i in range(len(indexes)):
        image_name = dir + dataframe.loc[indexes[i], 'image_name']
        im = cv2.imread(image_name)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols,i+1)
        plt.imshow(im)
        plt.title(labels[indexes[i]])
        plt.tight_layout()
    plt.show()

def create_label_shopping100k(df):
    df['category_name'] = df['image_name'].apply(lambda x: ' '.join(map(str, x.split('/')[1].split('_')[1:])))
    labels = df['category_name'].values.tolist()
    return labels