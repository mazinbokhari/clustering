from os import path
from matplotlib import pyplot as plt
from sklearn.mixture import GMM, VBGMM, DPGMM
import numpy as np

"""
INPUT
-----
image: an width-by-height-by-num_features numpy array that represents an image
n_components: the number of mixture components

OUTPUT
------
proba: an width-by-height-by-n_components array that represents the pmfs of each pixel
""" 
def get_pmfs(image, func=DPGMM, n_components=6):
    print("image shape:", image.shape)
    w, h, n_features = image.shape
    flat = image.reshape(w*h, n_features)
    model = func(n_components=n_components)
    model.fit(flat)
    proba = model.predict_proba(flat)
    proba.shape = (w, h, n_components)
    return proba

cmaps = ['OrRd', 'Greens', 'Blues', 'RdPu']

def plot_class(classprobs, c, output_dir="."):
    plt.title("Class {}".format(c))
    plt.imshow(classprobs, cmap=cmaps[c % len(cmaps)])
    plt.savefig(path.join(output_dir, "class_{}.png".format(c)))
    plt.close()

def plot_prob_classes(proba, output_dir="."):
    for c in range(proba.shape[-1]):
        plot_class(proba[:, :, c], c, output_dir=output_dir)

def plot_hard_classes(proba, output_dir):
    seg_img = np.argmax(proba, axis=-1)
    plt.title("Hard Classes")
    plt.imshow(seg_img, cmap=plt.cm.Paired)
    plt.savefig(path.join(output_dir, "hard_classes.png"))
    plt.close()
    
