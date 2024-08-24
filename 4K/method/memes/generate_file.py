import pickle
from rdkit import Chem
import numpy as np
from sklearn.cluster import KMeans
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

n_clusters = 40

data = dict()
ligand_list = list()
smiles_list = list()
feature_list = list()
clusters = dict()

all_smiles = load_pickle("./../data/ligand_smile.dat")

sentences = []
for ligand, smiles in all_smiles:
    mol = Chem.MolFromSmiles(smiles)
    #fingerprint = Chem.RDKFingerprint(mol)
    sentence = MolSentence(mol2alt_sentence(mol, 1))

    ligand_list.append(ligand)
    smiles_list.append(smiles)
    #feature_list.append(fingerprint)
    sentences.append(sentence)

model = word2vec.Word2Vec.load("model_300dim.pkl")
feature_list = [DfVec(x) for x in sentences2vec(sentences, model, unseen="UNK")]
feature_list = [x.vec for x in feature_list]
data["data"] = feature_list

# k-means clustering
clustering = KMeans(n_clusters, init="k-means++")
features = [ feature_list[ligand_list.index(ligand)] for ligand in ligand_list ]
clustering.fit(np.stack(features))

clusters = {}
for i, label in enumerate(clustering.labels_):
    clusters[ligand_list[i]] = label

save_pickle(clusters, "data/clusters.pkl")

#
with open("all.txt", "w") as f:
    for smiles in smiles_list:
        f.write(smiles+"\n")

with open(f"labels{n_clusters}.txt", "w") as f:
    for ligand, cluster in clusters.items():
        f.write(str(cluster)+"\n")

with open("ligands.txt", "w") as f:
    for ligand in ligand_list:
        f.write(ligand+"\n")

save_pickle(data, "features.pkl")
