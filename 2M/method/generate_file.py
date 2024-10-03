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

n_clusters = 2000

ligand_list = list()
smiles_list = list()
feature_list = list()
clusters = dict()

all_smiles = load_pickle("./../../data/smile_list.dat")

model = word2vec.Word2Vec.load("memes/data/model_300dim.pkl")
feature_list = np.memmap("features.npy", dtype=np.float16, mode='w+', shape=(2104318, 300))
for i, smiles in enumerate(all_smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        mol = Chem.MolFromSmiles(smiles)
        #fingerprint = Chem.RDKFingerprint(mol)
        sentence = MolSentence(mol2alt_sentence(mol, 1))
        del mol

        vec = sentences2vec([sentence], model, unseen="UNK")
        del sentence
        dfvec = DfVec(vec[0])
        feature = dfvec.vec
        del vec
        del dfvec

        #feature_list.append(fingerprint)
        #sentences.append(sentence)
        feature_list[i] = feature
        del feature
        
        # 리간드 파일명 대신 인덱스를 사용
        # ligand = f"ligand_{i}"
        
        # ligand_list.append(ligand)
        smiles_list.append(smiles)

feature_list.flush()
save_pickle(feature_list, "features.pkl")

# # k-means clustering
# clustering = KMeans(n_clusters, init="k-means++")
# features = [feature_list[ligand_list.index(ligand)] for ligand in ligand_list]
# clustering.fit(np.stack(features))

# clusters = {}
# for i, label in enumerate(clustering.labels_):
#     clusters[ligand_list[i]] = label

# save_pickle(clusters, "clusters.pkl")

# # SMILES 리스트를 파일에 저장
# with open("all.txt", "w") as f:
#     for smiles in smiles_list:
#         f.write(smiles+"\n")

# # 클러스터 레이블을 파일에 저장
# with open(f"labels{n_clusters}.txt", "w") as f:
#     for ligand, cluster in clusters.items():
#         f.write(str(cluster)+"\n")

# # 리간드 리스트를 파일에 저장
# with open("ligands.txt", "w") as f:
#     for ligand in ligand_list:
#         f.write(ligand+"\n")
