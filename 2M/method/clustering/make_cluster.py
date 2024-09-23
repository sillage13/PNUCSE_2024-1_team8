import os
import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle
from tqdm import tqdm

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

if __name__ == "__main__":
    # 데이터 로드
    smiles_list = load_data("./../../data/smile_list.dat")
    fingerprints = load_data("./../../data/fingerprints.dat")
    
    # 클러스터 수
    k_values = 1000
    
    print(f"Clustering with k={k_values}")
    
    # 클러스터링 수행
    labels = cluster_fingerprints(fingerprints, n_clusters=k_values)
    
    # 결과 저장
    output_file = f"labels{k_values}.txt"
    np.savetxt(output_file, labels, fmt='%d')
    
    print(f"Clustering results saved to {output_file}")