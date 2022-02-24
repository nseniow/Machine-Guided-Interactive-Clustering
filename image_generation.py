import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def generate_image(cluster_iter, numpy_data, labels, reduction_algorithm):

    if reduction_algorithm == "TSNE":
        tsne = TSNE()
        X_embedded = tsne.fit_transform(numpy_data)
        
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels)
        plt.gca().set_aspect('equal', 'datalim')
        plt.savefig("interactive-constrained-clustering/src/images/clusterImg" + cluster_iter, orientation='portrait')  # dpi=100 for landing page pic

    elif reduction_algorithm == "PCA":
        pca = PCA(n_components=2) 
        principalComponents = pca.fit_transform(numpy_data)
        principalDf = pd.DataFrame(data = principalComponents).to_numpy()

        plt.scatter(principalDf[:, 0], principalDf[:, 1], c=labels)
        plt.gca().set_aspect('equal', 'datalim')

        plt.savefig("interactive-constrained-clustering/src/images/clusterImg" + cluster_iter, orientation='portrait')  # dpi=100 for landing page pic

    elif reduction_algorithm == "UMAP":
        reducer = umap.UMAP()
        scaled_data = StandardScaler().fit_transform(numpy_data)
        embedding =  reducer.fit_transform(scaled_data)

        plt.scatter(embedding[:, 0], embedding[:, 1], c=labels)
        plt.gca().set_aspect('equal', 'datalim')
        #plt.title("BEANS", fontsize=24)
        plt.savefig("interactive-constrained-clustering/src/images/clusterImg" + cluster_iter, orientation='portrait')  # dpi=100 for landing page pic
    
    else:
        raise ValueError("Unknown algorithm")