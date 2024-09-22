import os
import pickle
from openbabel import pybel
from rdkit import Chem
from openbabel import openbabel


def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# PDBQT 파일을 RDKit Mol 객체로 변환하는 함수
def pdbqt_to_mol(pdbqt_file):
    mols = list(pybel.readfile("pdbqt", pdbqt_file))
    if mols:
        mol = mols[0]  # 첫 번째 분자만 사용
        pdb_block = mol.write("pdb")
        return Chem.MolFromPDBBlock(pdb_block)
    return None

# ./ligand 디렉토리 내의 모든 PDBQT 파일 읽기
ligand_dir = './ligand'
ligand_files = [f for f in os.listdir(ligand_dir) if f.endswith('.pdbqt')]
ligand_files.sort()

none_ligands = []
#모든 리간드를 RDKit Mol 객체로 변환하여 리스트에 저장
ligand_smiles = []
for i, ligand_file in enumerate(ligand_files):
    pdbqt_path = os.path.join(ligand_dir, ligand_file)

    mol = pdbqt_to_mol(pdbqt_path)
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        ligand_smiles.append((ligand_file, smiles))
    else:
        none_ligands.append(ligand_file)

print(none_ligands)
print(len(none_ligands))

for i in ligand_smiles:
    print(i)

# SMILES 리스트를 바이너리 파일로 저장
save_data(ligand_smiles, 'ligand_smile.dat')

print("SMILES data has been saved to ligand_smile.dat")

# 저장된 바이너리 파일을 불러오는 코드 예제
loaded_ligand_smiles = load_data('ligand_smile.dat')

print("Loaded SMILES from file:")
for i, smiles in enumerate(loaded_ligand_smiles):
    print(f"Ligand {i + 1}: {smiles}")

