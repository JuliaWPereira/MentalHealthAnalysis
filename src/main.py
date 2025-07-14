import sys
from utils.preprocess import preprocess_data
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

if __name__ == "__main__":
    # Parâmetros
    input_path = sys.argv[1] if len(sys.argv) > 1 else "../data/prepared_data.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "../data/processed_health_data.csv"
    n_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_runs = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    # Preprocessar dados uma vez
    df_processed = preprocess_data(input_path, output_path)

    times_kmeans = []
    times_gmm = []
    for i in range(n_runs):
        seed = np.random.randint(0, 10000)
        # KMeans
        start = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        #kmeans.fit(df_processed)
        times_kmeans.append(time.time() - start)
        # GMM
        start = time.time()
        gmm = GaussianMixture(n_components=n_clusters, random_state=seed)
        #gmm.fit(df_processed)
        times_gmm.append(time.time() - start)

    # Plotar comparação dos tempos
    plt.figure(figsize=(10,6))
    plt.plot(times_kmeans, label='KMeans')
    plt.plot(times_gmm, label='GMM')
    plt.xlabel('Execução')
    plt.ylabel('Tempo (s)')
    plt.title('Comparação dos Tempos de Execução: KMeans vs GMM')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Média KMeans:", np.mean(times_kmeans))
    print("Média GMM:", np.mean(times_gmm))
    print("Desvio Padrão KMeans:", np.std(times_kmeans))
    print("Desvio Padrão GMM:", np.std(times_gmm))
