from vina import Vina
import numpy as np
import random
import os
import pickle
from collections import defaultdict
import argparse


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
    args = parser.parse_args()
    receptor = args.receptor
    receptor_path = f"./../data/receptor/{receptor}.pdbqt"
    print(receptor_path)

    # 클러스터링 결과와 리간드 데이터 로드
    clustering = load_data('filtered_cluster.dat')
    ligands = load_data('./../data/ligand_smile.dat')
    docking_result = load_data(f'./../data/{receptor}_docking_result.dat')

    clusters = dict()
    for ligand, label in clustering:
        if label not in clusters:
            clusters[label] = list()
        clusters[label].append(ligand)

    vina_result = list()
    cluster_scores = dict()
    check_duplicate = defaultdict(int)
    cnt = 0
    for label, cluster_ligands in clusters.items():
        cnt += 1
        sample_ligands = random.sample(cluster_ligands, 2)

        print(f'cluster{label} {cnt}/{len(clusters)}(1/2)')
        print(f'{sample_ligands[0][0]}')
        if sample_ligands[0][0] not in docking_result:
            try:
                a = cal_score(f'./../data/ligand/{sample_ligands[0][0]}', receptor_path)
            except:
                a = 0
            docking_result[sample_ligands[0][0]] = a
        else:
            a = docking_result[sample_ligands[0][0]]
        vina_result.append([a, sample_ligands[0][0]])
        check_duplicate[sample_ligands[0][0]] = 1

        print(f'cluster{label} {cnt}/{len(clusters)}(2/2)')
        print(f'{sample_ligands[1][0]}')
        if sample_ligands[1][0] not in docking_result:
            try:
                b = cal_score(f'./../data/ligand/{sample_ligands[1][0]}', receptor_path)
            except:
                b = 0
            docking_result[sample_ligands[1][0]] = b
        else:
            b = docking_result[sample_ligands[1][0]]
        vina_result.append([b, sample_ligands[1][0]])
        check_duplicate[sample_ligands[1][0]] = 1

        cluster_scores[label] = (a + b) / 2
        print(f'cluster{label}\'s score: {cluster_scores[label]}')
        print()


    for i in range(10):
        highest_score_label = min(cluster_scores, key=cluster_scores.get)
        while True:
            sample_ligand = random.choice(clusters[highest_score_label])
            if check_duplicate[sample_ligand[0]] == 0:
                check_duplicate[sample_ligand[0]] = 1
                break
        
        print(f'Round {i+1}')
        print(f'Select cluster{highest_score_label}')
        print(f'{sample_ligand[0]}')
        if sample_ligand[0] not in docking_result:
            try:
                score = cal_score(f'./../data/ligand/{sample_ligand[0]}', receptor_path)
            except:
                score = 0
            docking_result[sample_ligand[0]] = score
        else:
            score = docking_result[sample_ligand[0]]
        vina_result.append([score, sample_ligand[0]])
        cluster_scores[highest_score_label] = (cluster_scores[highest_score_label] + score) / 2

    vina_result.sort(key=lambda x: x[0])
    top_ligands = vina_result[:10]

    print("Top10 ligands")
    for ligand in top_ligands:
        print(f"File: {ligand[1]}, Score: {ligand[0]}")
    print()
    scores = [ligand[0] for ligand in top_ligands]
    avg_score = sum(scores) / len(scores)
    print(f"Average score of top10 ligands: {avg_score: .2f}")

    save_data(vina_result, f"./../result/{receptor}/cluster_result4.dat")
    save_data(docking_result, f"./../data/{receptor}_docking_result.dat")
