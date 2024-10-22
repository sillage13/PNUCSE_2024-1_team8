
import os
import sys
import numpy as np
import django
import argparse
import random
import json
import time
from utils import cal_score, load_data, save_data


# Set up Django environment
sys.path.append("/app")  # Adjust to your Django project path
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "virtual_screening.settings")
django.setup()

from virtual_screening.models import Ligand

# 각 클러스터에서 2개의 무작위 SMILES에 대한 평균 점수를 계산하는 함수

def assign_cluster_scores(smile_list, score_dict, cluster_indices):
    global docking_result, receptor, is_demo
    cluster_scores = {}
    cluster_smiles = {}
    selected_smiles = set()  # 중복 방지를 위한 집합

    # SMILES를 각각의 클러스터에 매핑
    for i, smile in enumerate(smile_list):
        cluster = cluster_indices[i]
        if cluster not in cluster_smiles:
            cluster_smiles[cluster] = []
        cluster_smiles[cluster].append(smile)

    # 각 클러스터에서 무작위로 2개의 SMILES를 선택하고 평균 점수 할당
    for cluster, smiles in cluster_smiles.items():
        remaining_smiles = [s for s in smiles if s not in selected_smiles]

        if len(remaining_smiles) >= 2:
            random_smiles = random.sample(remaining_smiles, 2)
            # 중복 제거를 위해 선택된 SMILES 추가
            selected_smiles.update(random_smiles)
            if is_demo:
                avg_score = (score_dict[random_smiles[0]] +
                             score_dict[random_smiles[1]]) / 2
                docking_result.append(
                    (score_dict[random_smiles[0]], random_smiles[0]))
                docking_result.append(
                    (score_dict[random_smiles[1]], random_smiles[1]))
            else:
                score0 = cal_score(random_smiles[0], receptor)
                score1 = cal_score(random_smiles[1], receptor)
                docking_result.append((score0, random_smiles[0]))
                docking_result.append((score1, random_smiles[1]))
                avg_score = (score0 + score1) / 2

            cluster_scores[cluster] = avg_score
        else:
            # 클러스터에 2개 미만의 SMILES가 있을 경우 처리 
            for smile in remaining_smiles:
                selected_smiles.add(smile)
                if is_demo:
                    docking_result.append((score_dict[smile], smile))
                    cluster_scores[cluster] = score_dict[smile]
                else:
                    score = cal_score(smile, receptor)
                    docking_result.append((score, smile))
                    cluster_scores[cluster] = score

    print(selected_smiles)

    return cluster_scores, cluster_smiles, selected_smiles


def update_lowest_score_cluster(cluster_scores, cluster_smiles, score_dict, selected_smiles, target_count):
    global docking_result, receptor, is_demo
    while len(selected_smiles) < target_count:
        # 가장 낮은 점수를 가진 클러스터 찾기
        lowest_cluster = min(cluster_scores, key=cluster_scores.get)
        smiles = cluster_smiles.get(lowest_cluster, [])

        # 현재 선택된 리간드 수와 가장 낮은 점수의 클러스터 인덱스 출력
        print(f"선택된 리간드: {len(selected_smiles)}, 가장 낮은 클러스터: {str(lowest_cluster).zfill(4)}, 점수: {cluster_scores[lowest_cluster]:.2f}")

        remaining_smiles = [s for s in smiles if s not in selected_smiles]

        if len(remaining_smiles) < 2:
            for smile in remaining_smiles:
                selected_smiles.add(smile)
                if is_demo:
                    docking_result.append((score_dict[smile], smile))
                else:
                    score = cal_score(smile, receptor)
                    docking_result.append((score, smile))

            cluster_scores.pop(lowest_cluster, None)
            continue

        # 비어있는 클러스터나 이미 모두 선택된 클러스터 건너뛰기
        # if len(smiles) < 2:
        #     # 2개 미만의 SMILES가 있는 클러스터는 이미 선택되지 않은 것만 추가
        #     for smile in smiles:
        #         if smile not in selected_smiles:
        #             selected_smiles.append(smile)
        #     # 재선택 방지를 위해 클러스터 제거
        #     cluster_scores.pop(lowest_cluster, None)
        #     continue

        # 클러스터에서 무작위로 2개의 SMILES 선택
        random_smiles = random.sample(remaining_smiles, 2)
        selected_smiles.update(random_smiles)

        # Calculate new average score and update docking_result
        if is_demo:
            scores = [score_dict[smile] for smile in random_smiles]
            for smile, score in zip(random_smiles, scores):
                docking_result.append((score, smile))
        else:
            scores = []
            for smile in random_smiles:
                score = cal_score(smile, receptor)
                docking_result.append((score, smile))
                scores.append(score)
        new_avg_score = sum(scores) / len(scores)

        # Update the cluster score
        cluster_scores[lowest_cluster] = (cluster_scores[lowest_cluster] + new_avg_score) / 2
        
    return cluster_scores, selected_smiles

# # (리간드_스마일, 점수) 튜플 리스트를 생성하고 점수로 정렬하는 함수
# def create_sorted_ligand_list(selected_smiles, score_dict):
#     ligand_score_list = [(smile, score_dict[smile]) for smile in selected_smiles]
#     # 점수로 오름차순 정렬
#     sorted_ligand_list = sorted(ligand_score_list, key=lambda x: x[1])
#     return sorted_ligand_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docking script")
    parser.add_argument('--receptor', type=str,
                        help='Path to the receptor file')
    parser.add_argument('--count', type=str, help='Docking count')
    parser.add_argument('--is_demo', type=str, help='Is demo')
    parser.add_argument('--result_dir', type=str, help='Result directory')
    args = parser.parse_args()
    receptor = args.receptor
    count = int(args.count)
    is_demo = (args.is_demo == "True")
    result_dir = args.result_dir

    start_time = time.time()

    exist_cluster = [500, 1000, 2000, 4000, 8000]

    if is_demo:
        if count >= 2104318:
            print("Too large count")
            exit(1)
        score_dict = load_data("/screening/data/demo/smile_score_dict.dat")
        smile_list = load_data("/screening/data/demo/smile_list.dat")
        selected_cluster = min(exist_cluster, key=lambda x: abs(x - count/4))
        cluster_indices = np.loadtxt(
            f"/screening/data/demo/labels{selected_cluster}.txt")

    else:
        score_dict = dict()
        smile_list = list(Ligand.objects.values_list(
            'ligand_smile', flat=True))
        selected_cluster = min(exist_cluster, key=lambda x: abs(x - count/4))
        cluster_indices = np.loadtxt(
            f"/screening/data/labels{selected_cluster}.txt")

    if len(smile_list) != len(cluster_indices):
        print(
            "Number of labels does not match number of ligands. "
            "Please create cluster files from the Manage Ligand page and try again."
        )
        exit(1)

    docking_result = list()

    # 클러스터 별로 2개의 SMILES를 선택하고 초기 점수를 할당
    cluster_scores, cluster_smiles, selected_smiles = assign_cluster_scores(
        smile_list, score_dict, cluster_indices)
    # 가장 점수가 낮은 클러스터의 점수를 반복 갱신
    updated_cluster_scores, final_selected_smiles = update_lowest_score_cluster(
        cluster_scores, cluster_smiles, score_dict, selected_smiles, count)

    # 최종 선택된 SMILES를 (스코어, 리간드 스마일) 형식으로 정렬
    docking_result.sort(key=lambda x: x[0])
    top_ligands = docking_result[:10]

    print("Top10 ligands")
    for ligand in top_ligands:
        print(f"File: {ligand[1]}, Score: {ligand[0]}")
    print()
    scores = [ligand[0] for ligand in top_ligands]
    avg_score = sum(scores) / len(scores)
    print(f"Average score of top10 ligands: {avg_score: .2f}")

    end_time = time.time()
    execution_time = end_time - start_time

    output_data = {
        'top_ligands': [{'smile': ligand[1], 'score': ligand[0]} for ligand in top_ligands],
        'avg_score': avg_score,
        'execution_time': execution_time
    }
    output_file = os.path.join(result_dir, 'result.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
