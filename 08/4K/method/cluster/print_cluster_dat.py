import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 클러스터링 결과와 리간드 데이터 로드
clustering = load_data('filtered_cluster.dat')
ligands = load_data('./../data/ligand_smile.dat')

# 클러스터 레이블 가져오기
# labels = clustering.labels_

# # 각 클러스터별로 리간드 그룹화
# clusters = {}
# for i, label in enumerate(labels):
#     if label not in clusters:
#         clusters[label] = []
#     clusters[label].append(ligands[i])

# size = list()
# # 결과 출력
# for label, cluster_ligands in clusters.items():
#     print(f"Cluster {label}:")
#     for ligand in cluster_ligands[:5]:  # 각 클러스터의 처음 5개 리간드만 출력
#         print(f"  {ligand[0]}: {ligand[1]}")
#     print(f"  ... (총 {len(cluster_ligands)}개 리간드)")
#     print()

for i in clustering:
    print(i)

# 각 클러스터별로 리간드 그룹화
clusters = dict()
for ligand, label in clustering:
    if label not in clusters:
        clusters[label] = list()
    clusters[label].append(ligand)

# 결과 출력
for label, cluster_ligands in clusters.items():
    print(f"Cluster {label}:")
    for ligand in cluster_ligands[:5]:  # 각 클러스터의 처음 5개 리간드만 출력
        print(f"  {ligand[0]}: {ligand[1]}")
    print(f"  ... (총 {len(cluster_ligands)}개 리간드)")
    print()

print(len(clusters))