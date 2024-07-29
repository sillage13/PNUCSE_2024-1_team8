import os
import pickle
from openbabel import pybel
from rdkit import Chem
from openbabel import openbabel
from vina import Vina
import numpy as np
import argparse


def cal_score(ligand, receptor):
    v = Vina(sf_name='vina')
    v.set_receptor(receptor)
    v.set_ligand_from_file(ligand)

    center_x, center_y, center_z = calculate_center_of_mass(ligand)
    v.compute_vina_maps(center=[center_x, center_y, center_z], box_size=[20, 20, 20])

    v.dock(exhaustiveness=32, n_poses=10)
    print()
    energy = v.score()
    
    return energy[0] 

def calculate_center_of_mass(pdbqt_file):
    x_coords = []
    y_coords = []
    z_coords = []

    with open(pdbqt_file, 'r') as file:
        for line in file:
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

def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docking script")
    parser.add_argument('receptor', type=str, help='Path to the receptor file')
    args = parser.parse_args()
    receptor = args.receptor
    receptor_path = f"./../receptor/{receptor}.pdbqt"
    print(receptor_path)

    tmp_result = load_data(f'scoring_{receptor}_20.dat')
    tmp_result.sort(key=lambda x: x[0])
    
    docking_result = load_data(f'./../data/{receptor}_docking_result.dat')

    candidate = tmp_result[:40]
    ligand_dir = './../data/ligand'
    vina_result = list()
    # for i, ligand_file in candidate:
    for i, v in enumerate(candidate):
        print(f"{v[1]}, {i+1}/{len(candidate)}")
        ligand_file = v[1]
        ligand_path = os.path.join(ligand_dir, ligand_file)
        if ligand_file not in docking_result:
            score = cal_score(ligand_path, receptor_path)
            docking_result[ligand_file] = score
        else:
            score = docking_result[ligand_file]
        vina_result.append((score, ligand_file))
    
    vina_result.sort(key=lambda x: x[0])
    top_ligands = vina_result[:10]

    print("Top10 ligands")
    for ligand in top_ligands:
        print(f"File: {ligand[1]}, Score: {ligand[0]}")
    print()
    scores = [ligand[0] for ligand in top_ligands]
    avg_score = sum(scores) / len(scores)
    print(f"Average score of top10 ligands: {avg_score: .2f}")
    save_data(vina_result, f'./../result/{receptor}/scoring_result_20.dat')
    save_data(docking_result, f'./../data/{receptor}_docking_result.dat')
