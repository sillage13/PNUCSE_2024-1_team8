import os
import pickle
from openbabel import pybel
from rdkit import Chem
from openbabel import openbabel
from vina import Vina
import numpy as np
import argparse
import pandas as pd
from rdkit.Chem import AllChem, DataStructs
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import dask.dataframe as dd


def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        arr = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return np.zeros((1024,), dtype=np.int8)  # 실패 시 0 벡터 반환
    
def plot_dendrogram(Z, max_clusters, labels, filename):
    plt.figure(figsize=(20, 10))  # 그림 크기 조정
    sch.dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=6, truncate_mode='lastp', p=max_clusters)
    plt.title(f'Dendrogram - {max_clusters} clusters')
    plt.xlabel('Ligand')
    plt.ylabel('Distance')
    plt.savefig(filename, dpi=150)  # 해상도 150dpi로 저장
    plt.close()


if __name__ == "__main__":
    smiles_list = load_data("./../../../data/smile_list.dat")

    # 모든 리간드에 대해 분자 특성 벡터 생성
    # fingerprints = np.array([smiles_to_fingerprint(smiles) for smiles in smiles_list])
    fingerprints = load_data("fingerprints.dat")

    # 계층적 클러스터링 수행
    fingerprints = fingerprints.astype(np.float32)
    dask_df = dd.from_pandas(pd.DataFrame(fingerprints), npartitions=10)

    Z = sch.linkage(dask_df, method='ward')

    save_data(Z, "hierarchical_clustering_matrix.dat")
    # # 덴드로그램 그리기 및 파일로 저장
    # plt.figure(figsize=(50, 10))  # 그림 크기 조정
    # dendrogram = sch.dendrogram(Z, labels=smiles_list, leaf_rotation=90, leaf_font_size=2)  # leaf_font_size 축소
    # plt.title('Dendrogram')
    # plt.xlabel('Ligand')
    # plt.ylabel('Distance')
    # plt.savefig('dendrogram.png', dpi=150)  # 해상도 150dpi로 저장
    # plt.close()

    cluster = defaultdict(list)

    # 클러스터 개수에 따라 덴드로그램 점진적으로 자르기
    for cnt in [16, 64, 256, 1024, 4096, 16384]:
        filename = f'dendrogram_{cnt}_clusters.png'
        plot_dendrogram(Z, cnt, smiles_list, filename)
        index = sch.cut_tree(Z, n_clusters=cnt).flatten()
        # print(f"{cnt}개의 클러스터 라벨:", clusters)
        for i, v in enumerate(smiles_list):
            cluster[v].append(index[i])
    
    save_data(cluster, "cluster.dat")

    # # memes에 사용될 클러스터 txt파일 생성
    # cnt = 10
    # filename = "labels10.txt"
    # index = sch.cut_tree(Z, n_clusters=cnt).flatten()
    # with open(filename, 'w') as file:
    #     for idx in index:
    #         file.write(f"{idx}\n")
        
