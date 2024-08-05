import argparse
from vina import Vina
import numpy as np
import random
import os
import pickle
from collections import defaultdict


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
    
def cal_score(ligand, receptor):
    v = Vina(sf_name='vina')
    v.set_receptor(receptor)
    v.set_ligand_from_file(ligand)

    center_x, center_y, center_z = calculate_center_of_mass(ligand)
    v.compute_vina_maps(center=[center_x, center_y, center_z], box_size=[120, 120, 120])

    v.dock(exhaustiveness=32, n_poses=10)
    print()
    energy = v.score()
    
    return energy[0] 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docking script")
    parser.add_argument('receptor', type=str, help='Path to the receptor file')
    parser.add_argument('count', type=str, help='Docking count')
    parser.add_argument('result_tail', type=str, help='result_tail')
    args = parser.parse_args()
    receptor = args.receptor
    count = int(args.count)
    result_tail = int(args.result_tail)
    receptor_path = f"./../data/receptor/{receptor}.pdbqt"
    print(receptor_path)

    ligands = load_data('./../data/ligand_smile.dat')
    docking_result = load_data(f'./../data/{receptor}_docking_result.dat')

    ligand_dir = './../data/ligand'
    random_result = list()
    random_ligands = random.sample(ligands, min(count, 4250))
    index = 1
    for ligand_file, _ in random_ligands:
        try:
            print(f"{ligand_file}, {index}/{count}")
            ligand_path = os.path.join(ligand_dir, ligand_file)
            if ligand_file not in docking_result:
                score = cal_score(ligand_path, receptor_path)
                docking_result[ligand_file] = score
            else:
                score = docking_result[ligand_file]
                print(score)
            random_result.append((score, ligand_file))
            index += 1
            # if index == count + 1:
            #     break
        except Exception as e:
            print(f"Error processing {ligand_file}: {e}")
            continue

    random_result.sort(key=lambda x: x[0])
    top_ligands = random_result[:10]

    print("Top10 ligands")
    for ligand in top_ligands:
        print(f"File: {ligand[1]}, Score: {ligand[0]}")
    print()
    scores = [ligand[0] for ligand in top_ligands]
    avg_score = sum(scores) / len(scores)
    print(f"Average score of top10 ligands: {avg_score: .2f}")

    save_data(random_result, f"./../result/{receptor}/{count}/random_result{result_tail}.dat")
    save_data(docking_result, f"./../data/{receptor}_docking_result.dat")
