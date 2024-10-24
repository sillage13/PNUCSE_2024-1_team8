import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
import argparse

def calculate_similarity_rdkit(smiles1, smiles2, fp_type='maccs'):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if fp_type == 'maccs':
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
    elif fp_type == 'topological':
        fp1 = AllChem.RDKFingerprint(mol1)
        fp2 = AllChem.RDKFingerprint(mol2)
    
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def calculate_similarity_mcs(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    
    similarity = (mcs.numAtoms + mcs.numBonds) / (mol1.GetNumAtoms() + mol1.GetNumBonds() + mol2.GetNumAtoms() + mol2.GetNumBonds())
    return similarity


def calculate_similarity_scaffolds(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    scaffold1 = Chem.MurckoDecompose(mol1)
    scaffold2 = Chem.MurckoDecompose(mol2)
    
    return scaffold1 == scaffold2


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def main(result_filename):
    smile_dict_filename = './../data/smile_dict.dat'
    smile_dict = load_data(smile_dict_filename)
    tmp = load_data(result_filename)
    result = tmp[:10]
    print(result)
    print(len(result))
    similarity = []
    msc = []
    scaffolds = []
    # 클러스터 결과 유사도 계산
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            smiles1 = smile_dict[result[i][1]]
            smiles2 = smile_dict[result[j][1]]
            # print(1)
            similarity.append(calculate_similarity_rdkit(smiles1, smiles2))
            # print(2)
            msc.append(calculate_similarity_mcs(smiles1, smiles2))
            # print(3)
            # scaffolds.append(calculate_similarity_scaffolds(smiles1, smiles2))
            # print("WORK")

    print("분자 지문 기반 유사도")
    print(f"Maximum similarity: {max(similarity):.2f}")
    print(f"Average similarity: {sum(similarity) / len(similarity):.2f}")
    print(f"Minimum similarity: {min(similarity):.2f}")

    print("최대 공통 부분구조 기반 유사도")
    print(f"Maximum similarity: {max(msc):.2f}")
    print(f"Average similarity: {sum(msc) / len(msc):.2f}")
    print(f"Minimum similarity: {min(msc):.2f}")

    # print("구조 키 기반 유사도")
    # print(f"Average similarity: {sum(scaffolds) / len(scaffolds):.2f}")
    # print(f"Maximum similarity: {max(scaffolds):.2f}")
    # print(f"Minimum similarity: {min(scaffolds):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate molecular similarities.')
    parser.add_argument('cluster_result_filename', type=str, help='The filename of the cluster result data file.')

    args = parser.parse_args()
    main(args.cluster_result_filename)
