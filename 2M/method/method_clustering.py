import os
import sys
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

# labels{n}.txt에서 클러스터 인덱스를 로드하는 함수
def load_cluster_indices(filename):
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]

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
        if len(smiles) >= 2:
            random_smiles = random.sample(smiles, 2)
            # 중복 제거를 위해 선택된 SMILES 추가
            selected_smiles.update(random_smiles)
            if is_demo:
                avg_score = (score_dict[random_smiles[0]] + score_dict[random_smiles[1]]) / 2
                docking_result.append((score_dict[random_smiles[0]], random_smiles[0]))
                docking_result.append((score_dict[random_smiles[1]], random_smiles[1]))
            else:
                score0 = cal_score(random_smiles[0], receptor)
                score1 = cal_score(random_smiles[1], receptor)
                docking_result.append((score0, random_smiles[0]))
                docking_result.append((score1, random_smiles[1]))
                avg_score = (score0 + score1) / 2

            cluster_scores[cluster] = avg_score
        else:
            # 클러스터에 2개 미만의 SMILES가 있을 경우 처리
            selected_smiles.update(smiles)
            if is_demo:
                if smiles:
                    cluster_scores[cluster] = score_dict[smiles[0]]
                    docking_result.append((score_dict[smiles[0]], smiles[0]))
                else:
                    cluster_scores[cluster] = 0
            else:
                if smiles:
                    cluster_scores[cluster] = cal_score(smiles[0], receptor)
                    docking_result.append((cluster_scores[cluster], smiles[0]))
                else:
                    cluster_scores[cluster] = 0

    return cluster_scores, cluster_smiles, list(selected_smiles)

def update_lowest_score_cluster(cluster_scores, cluster_smiles, score_dict, selected_smiles, target_count):
    global docking_result, receptor, is_demo
    while len(selected_smiles) < target_count:
        # 가장 낮은 점수를 가진 클러스터 찾기
        lowest_cluster = min(cluster_scores, key=cluster_scores.get)
        smiles = cluster_smiles.get(lowest_cluster, [])

        # 현재 선택된 리간드 수와 가장 낮은 점수의 클러스터 인덱스 출력
        print(f"선택된 리간드: {len(selected_smiles)}, 가장 낮은 클러스터: {str(lowest_cluster).zfill(4)}, 점수: {cluster_scores[lowest_cluster]:.2f}")

        # 비어있는 클러스터나 이미 모두 선택된 클러스터 건너뛰기
        if len(smiles) < 2:
            # 2개 미만의 SMILES가 있는 클러스터는 이미 선택되지 않은 것만 추가
            for smile in smiles:
                if smile not in selected_smiles:
                    selected_smiles.append(smile)
            # 재선택 방지를 위해 클러스터 제거
            cluster_scores.pop(lowest_cluster, None)
            continue

        # 클러스터에서 무작위로 2개의 SMILES 선택
        random_smiles = random.sample(smiles, 2)

        # 중복을 피하고 selected_smiles에 추가
        added = False
        for smile in random_smiles:
            if smile not in selected_smiles:
                selected_smiles.append(smile)
                added = True

        # 새로운 SMILES가 추가된 경우에만 클러스터 점수 업데이트
        if added:
            if is_demo:
                new_avg_score = (score_dict[random_smiles[0]] + score_dict[random_smiles[1]]) / 2
                docking_result.append((score_dict[random_smiles[0]], random_smiles[0]))
                docking_result.append((score_dict[random_smiles[1]], random_smiles[1]))
            else:
                score0 = cal_score(random_smiles[0], receptor)
                score1 = cal_score(random_smiles[1], receptor)
                docking_result.append((score0, random_smiles[0]))
                docking_result.append((score1, random_smiles[1]))
                new_avg_score = (score0 + score1) / 2
            cluster_scores[lowest_cluster] = (cluster_scores[lowest_cluster] + new_avg_score) / 2
        else:
            # 새로운 SMILES가 추가되지 않았다면 다음 루프에서 이 클러스터 건너뛰기
            cluster_scores.pop(lowest_cluster, None)

    return cluster_scores, selected_smiles

# # (리간드_스마일, 점수) 튜플 리스트를 생성하고 점수로 정렬하는 함수
# def create_sorted_ligand_list(selected_smiles, score_dict):
#     ligand_score_list = [(smile, score_dict[smile]) for smile in selected_smiles]
#     # 점수로 오름차순 정렬
#     sorted_ligand_list = sorted(ligand_score_list, key=lambda x: x[1])
#     return sorted_ligand_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docking script")
    parser.add_argument('--receptor', type=str, help='Path to the receptor file')
    parser.add_argument('--count', type=str, help='Docking count')
    parser.add_argument('--is_demo', type=str, help='Is demo')
    parser.add_argument('--result_dir', type=str, help='Result directory')
    args = parser.parse_args()
    receptor = args.receptor
    count = int(args.count)
    is_demo = (args.is_demo == "True")
    result_dir = args.result_dir
    
    start_time = time.time()


    if is_demo:
        if count >= 2104318:
            print("Too large count")
            exit(1)
        score_dict = load_data("/screening/data/demo/smile_score_dict.dat")
        smile_list = load_data("/screening/data/demo/smile_list.dat")
        exist_cluster = [500, 1000, 2000, 4000, 8000]
        selected_cluster = min(exist_cluster, key=lambda x: abs(x - count/4))
        cluster_indices = load_cluster_indices(f"/screening/data/demo/labels{selected_cluster}.txt")

    else:
        smile_list = list(Ligand.objects.values_list('ligand_smile', flat=True))
        #TODO 데이터 연결
        # 클러스터 파일 선택
        # 
    

    docking_result = list()

    # 클러스터 별로 2개의 SMILES를 선택하고 초기 점수를 할당
    cluster_scores, cluster_smiles, selected_smiles = assign_cluster_scores(smile_list, score_dict, cluster_indices)
    # 가장 점수가 낮은 클러스터의 점수를 반복 갱신
    updated_cluster_scores, final_selected_smiles = update_lowest_score_cluster(cluster_scores, cluster_smiles, score_dict, selected_smiles, count)

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
