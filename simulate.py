import sklearn
import numpy as np

def generate_images(image, n_clusters=6):
    model = sklearn.cluster.KMeans(n_clusters=n_clusters)
    model.fit(image)
    values = k_means.cluster_centers_.squeeze()
    labels = k_means.labels_
    true_image = np.choose(labels, values)
    true_image.shape = image.shape
    print(labels)
