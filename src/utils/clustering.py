from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import pandas as pd

def run_kmeans(data_path, n_clusters=3, random_state=42):
    """
    Executa o KMeans e retorna os labels e o modelo.
    data_path: caminho para o arquivo csv com dados normalizados
    n_clusters: número de clusters
    """
    df = pd.read_csv(data_path)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(df)
    return labels, kmeans

def run_gmm(data_path, n_components=3, random_state=42):
    """
    Executa o GaussianMixture (GMM) e retorna os labels e o modelo.
    data_path: caminho para o arquivo csv com dados normalizados
    n_components: número de componentes (clusters)
    """
    df = pd.read_csv(data_path)
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(df)
    return labels, gmm