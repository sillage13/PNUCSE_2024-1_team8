import pickle
from vina import Vina
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
import os


def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)
        

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
    
def smiles_to_pdbqt(smiles):
    # SMILES를 3D 구조로 변환
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)

    # RDKit 분자를 PDB 문자열로 변환
    pdb = Chem.MolToPDBBlock(mol)

    # PDB를 PDBQT로 변환
    pdb_mol = pybel.readstring("pdb", pdb)
    pdb_mol.addh()
    pdb_mol.make3D()
    pdbqt = pdb_mol.write("pdbqt")

    return pdbqt


def smiles_to_fingerprint(smiles):
    fpGenerator = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, countSimulation=False)
    
    return fpGenerator.GetFingerprint(Chem.MolFromSmiles(smiles)).ToBitString()


def calculate_center_of_mass(pdbqt_file):
    x_coords = []
    y_coords = []
    z_coords = []
    
    for line in pdbqt_file.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            x_coords.append(x)
            y_coords.append(y)
            z_coords.append(z)
            
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    center_z = sum(z_coords) / len(z_coords)
    
    return center_x, center_y, center_z
    
def cal_score(smile, receptor):
    try:
        ligand = smiles_to_pdbqt(smile)
        v = Vina(sf_name='vina')
        v.set_receptor(receptor)
        v.set_ligand_from_string(ligand)

        center_x, center_y, center_z = calculate_center_of_mass(ligand)
        v.compute_vina_maps(center=[center_x, center_y, center_z], box_size=[60, 60, 60])

        v.dock(exhaustiveness=32, n_poses=10)
        print()
        energy = v.score()
    
    except Exception as e:
        print(f"Error processing {smile}: {e}")
        return None
    
    return energy[0] 

def save_output_xyz(smile, ligandID, receptor, dirPath):
    try:
        ligand = smiles_to_pdbqt(smile)
        v = Vina(sf_name='vina')
        v.set_receptor(os.path.join(dirPath, receptor))
        v.set_ligand_from_string(ligand)

        center_x, center_y, center_z = calculate_center_of_mass(ligand)
        v.compute_vina_maps(center=[center_x, center_y, center_z], box_size=[60, 60, 60])

        v.dock(exhaustiveness=32, n_poses=10)
        pose = v.poses(n_poses=1)

        mol = pybel.readstring("pdbqt", pose)
        mol.write("xyz", os.path.join(dirPath, f"output{ligandID}.xyz"), overwrite=True)
    
    except Exception as e:
        print(f"Error processing {smile}: {e}")
        return None
    
def save_receptor_pdb(receptor, dirPath):
    try:
        mol = next(pybel.readfile("pdbqt", os.path.join(dirPath, receptor)))
        mol.write("pdb", os.path.join(dirPath, "receptor.pdb"), overwrite=True)

    except Exception as e:
        print(f"Error processing {receptor}: {e}")
        return None