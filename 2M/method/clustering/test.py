import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt


def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def cluster_fingerprints(fingerprints, n_clusters, batch_size=10000, random_state=42):
    n_samples, n_features = fingerprints.shape
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=batch_size)
    
    # 데이터 셔플
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # 초기 피팅
    initial_batch = fingerprints[indices[:batch_size]]
    kmeans.partial_fit(initial_batch)
    
    # 나머지 데이터에 대해 부분 피팅
    for start in tqdm(range(batch_size, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch = fingerprints[indices[start:end]]
        kmeans.partial_fit(batch)
    
    # 최종 예측
    labels = np.zeros(n_samples, dtype=int)
    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch = fingerprints[start:end]
        labels[start:end] = kmeans.predict(batch)
    
    return labels


# def plot_cluster_distribution(labels, n_clusters, output_file):
#     plt.figure(figsize=(12, 6))
#     plt.hist(labels, bins=n_clusters, range=(0, n_clusters), align='left', rwidth=0.8)
#     plt.title(f'Distribution of Cluster Indices (k={n_clusters})')
#     plt.xlabel('Cluster Index')
#     plt.ylabel('Number of Samples')
#     plt.savefig(output_file)
#     plt.close()


if __name__ == "__main__":
    # 데이터 로드
    smiles_list = load_data("./../../../data/smile_list.dat")
    fingerprints = load_data("fingerprints.dat")
    
    # 클러스터 수 리스트
    k_values = [500, 1000, 2000, 4000, 8000]
    
    for k in k_values:
        print(f"Clustering with k={k}")
        
        # 클러스터링 수행
        labels = cluster_fingerprints(fingerprints, n_clusters=k)
        
        # 결과 저장
        output_file = f"labels{k}.txt"
        np.savetxt(output_file, labels, fmt='%d')
        
        print(f"Clustering results saved to {output_file}")
        
        # 클러스터 분포 플롯 생성 및 저장
        # plot_file = f"cluster_distribution_{k}.png"
        # plot_cluster_distribution(labels, k, plot_file)
        # print(f"Cluster distribution plot saved to {plot_file}")