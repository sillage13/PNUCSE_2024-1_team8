import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFMCS
import re
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
    

def extract_numbers_from_input():
    numbers = []
    print("입력")
    for _ in range(10):
        # 사용자로부터 입력을 받음
        line = input()
        # 정규 표현식을 사용하여 ligand 뒤의 숫자만 추출
        match = re.search(r'ligand(\d+)', line)
        if match:
            numbers.append(int(match.group(1)))  # 추출된 숫자를 정수로 변환하여 리스트에 추가
    return numbers


if __name__ == "__main__":
    index_list = extract_numbers_from_input()
    smile_iist = load_data('./../data/smile_list.dat')

    print(index_list)
    print(len(index_list))
    similarity = []
    msc = []
    scaffolds = []
    # 클러스터 결과 유사도 계산
    for i in range(len(index_list)):
        for j in range(i + 1, len(index_list)):
            smiles1 = smile_iist[index_list[i]]
            smiles2 = smile_iist[index_list[j]]
            # print(1)
            similarity.append(calculate_similarity_rdkit(smiles1, smiles2))
            print(similarity[-1])
            # print(2)
            # msc.append(calculate_similarity_mcs(smiles1, smiles2))
            # print(3)
            # scaffolds.append(calculate_similarity_scaffolds(smiles1, smiles2))
            # print("WORK")

    print("분자 지문 기반 유사도")
    print(f"Maximum similarity: {max(similarity):.2f}")
    print(f"Average similarity: {sum(similarity) / len(similarity):.2f}")
    print(f"Minimum similarity: {min(similarity):.2f}")

    # print("최대 공통 부분구조 기반 유사도")
    # print(f"Maximum similarity: {max(msc):.2f}")
    # print(f"Average similarity: {sum(msc) / len(msc):.2f}")
    # print(f"Minimum similarity: {min(msc):.2f}")

    # print("구조 키 기반 유사도")
    # print(f"Average similarity: {sum(scaffolds) / len(scaffolds):.2f}")
    # print(f"Maximum similarity: {max(scaffolds):.2f}")
    # print(f"Minimum similarity: {min(scaffolds):.2f}")



