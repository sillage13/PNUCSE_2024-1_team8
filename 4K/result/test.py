import os
import sys
from openbabel import openbabel
import pickle

def pdbqt_to_smiles(pdbqt_file):
    # OBConversion 객체 생성 및 입력/출력 형식 설정
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat("pdbqt")  # 입력 형식을 pdbqt로 설정
    obConversion.SetOutFormat("smi")   # 출력 형식을 SMILES로 설정

    # OBMol 객체 생성
    mol = openbabel.OBMol()

    # PDBQT 파일 읽기
    if not obConversion.ReadFile(mol, pdbqt_file):
        print(f"Error: Could not read the PDBQT file {pdbqt_file}")
        return None

    # SMILES 문자열을 한 줄씩 읽고 출력
    smiles = obConversion.WriteString(mol).strip()  # SMILES 문자열 생성
    return smiles.split()[0] if smiles else None

def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # directory = "./../ligand"
    # ligand_dict = {}

    # for filename in os.listdir(directory):
    #     if filename.endswith(".pdbqt"):
    #         filepath = os.path.join(directory, filename)
    #         smiles = pdbqt_to_smiles(filepath)
    #         if smiles:
    #             ligand_dict[filename] = smiles

    # save_data(ligand_dict, "smile_dict2.dat")
    a = load_data("smile_dict2.dat")
    for i, v in a.items():
        print(i, v)
