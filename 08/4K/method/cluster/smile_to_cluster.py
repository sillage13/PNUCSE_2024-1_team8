import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from rdkit import DataStructs
import pickle

def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

# Load ligands
ligands = load_data('./../data/ligand_smile.dat')
print(len(ligands))

# 2. Parse chemical structures and calculate features (Morgan fingerprint)
mols = [Chem.MolFromSmiles(smiles[1]) for smiles in ligands]
fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]

# 3. Create similarity matrix
fp_array = np.array([np.array(fp) for fp in fps])
similarity_matrix = 1 - pairwise_distances(fp_array, metric='jaccard')

# 4. Apply clustering algorithm (Agglomerative Clustering)
initial_n_clusters = 200
clustering = AgglomerativeClustering(n_clusters=initial_n_clusters, metric='precomputed', linkage='complete')
clustering.fit(1 - similarity_matrix)

# Get cluster labels
labels = clustering.labels_

# Group ligands by cluster
clusters = {}
for i, label in enumerate(labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(ligands[i])

# Ensure each cluster has at least 50 ligands
min_ligands_per_cluster = 50
filtered_clusters = {}
small_clusters = []

for label, cluster_ligands in clusters.items():
    if len(cluster_ligands) >= min_ligands_per_cluster:
        filtered_clusters[label] = cluster_ligands
    else:
        small_clusters.extend(cluster_ligands)

# If there are small clusters, merge them into other clusters
for ligand in small_clusters:
    # Find the closest large cluster
    best_cluster = None
    best_similarity = -1
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ligand[1]), 2, nBits=1024)
    for label, cluster_ligands in filtered_clusters.items():
        cluster_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(cluster_ligands[0][1]), 2, nBits=1024)
        similarity = tanimoto_similarity(fp, cluster_fp)
        if similarity > best_similarity:
            best_similarity = similarity
            best_cluster = label
    filtered_clusters[best_cluster].append(ligand)

# Save the new clustering result
new_labels = []
for label, cluster_ligands in filtered_clusters.items():
    for ligand in cluster_ligands:
        new_labels.append((ligand, label))

save_data(new_labels, 'filtered_cluster.dat')

# Print the sizes of the clusters
for label, cluster_ligands in filtered_clusters.items():
    print(f"Cluster {label}: {len(cluster_ligands)} ligands")

# Optional: Re-run clustering if the cluster sizes are still not balanced
