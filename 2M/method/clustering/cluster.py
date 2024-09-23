import random
import pickle

# 나중에 사용할 저장/로드 함수
def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# labels{n}.txt에서 클러스터 인덱스를 로드하는 함수
def load_cluster_indices(filename):
    with open(filename, 'r') as f:
        return [int(line.strip()) for line in f]

# 각 클러스터에서 2개의 무작위 SMILES에 대한 평균 점수를 계산하는 함수
def assign_cluster_scores(smile_list, score_dict, cluster_indices):
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
            avg_score = (score_dict[random_smiles[0]] + score_dict[random_smiles[1]]) / 2
            cluster_scores[cluster] = avg_score
        else:
            # 클러스터에 2개 미만의 SMILES가 있을 경우 처리
            selected_smiles.update(smiles)
            cluster_scores[cluster] = score_dict[smiles[0]] if smiles else 0

    return cluster_scores, cluster_smiles, list(selected_smiles)

def update_lowest_score_cluster(cluster_scores, cluster_smiles, score_dict, selected_smiles, target_count=21000):
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
            new_avg_score = (score_dict[random_smiles[0]] + score_dict[random_smiles[1]]) / 2
            cluster_scores[lowest_cluster] = new_avg_score
        else:
            # 새로운 SMILES가 추가되지 않았다면 다음 루프에서 이 클러스터 건너뛰기
            cluster_scores.pop(lowest_cluster, None)

    return cluster_scores, selected_smiles

# (리간드_스마일, 점수) 튜플 리스트를 생성하고 점수로 정렬하는 함수
def create_sorted_ligand_list(selected_smiles, score_dict):
    ligand_score_list = [(smile, score_dict[smile]) for smile in selected_smiles]
    # 점수로 오름차순 정렬
    sorted_ligand_list = sorted(ligand_score_list, key=lambda x: x[1])
    return sorted_ligand_list

if __name__ == "__main__":
    # 데이터 로드
    score_dict = load_data("./../../data/smile_score_dict.dat")
    smile_list = load_data("./../../data/smile_list.dat")
    smile_lidex = load_data("./../../data/smile_index_dict.dat")
    cluster_indices = load_cluster_indices("labels8000.txt")

    # 클러스터 별로 2개의 SMILES를 선택하고 초기 점수를 할당
    cluster_scores, cluster_smiles, selected_smiles = assign_cluster_scores(smile_list, score_dict, cluster_indices)
    # 가장 점수가 낮은 클러스터의 점수를 2500번 갱신
    updated_cluster_scores, final_selected_smiles = update_lowest_score_cluster(cluster_scores, cluster_smiles, score_dict, selected_smiles)

    # 최종 선택된 SMILES를 (리간드 스마일, 스코어) 형식으로 정렬
    # 점수로 정렬하되, 점수가 같으면 인덱스가 낮은 순서로 정렬
    sorted_ligands = sorted(
        create_sorted_ligand_list(final_selected_smiles, score_dict),
        key=lambda x: (x[1], smile_lidex[x[0]])  # 점수 -> 인덱스 순서로 정렬
    )

    # 가장 낮은 점수를 갖는 10개의 리간드 출력
    print("가장 낮은 점수를 갖는 10개의 리간드:")
    tmp = 0
    for ligand, score in sorted_ligands[:10]:
        print(f"{smile_lidex[ligand]}, {score}")
        tmp += score
    print(f"상위 10개 리간드의 평균 점수: {tmp / 10:.2f}")

    # 선택된 리간드의 총 개수 출력
    print(f"총 선택된 리간드: {len(sorted_ligands)}")

    # 전체 선택된 리간드의 점수 평균 계산
    scores = [score for _, score in sorted_ligands]  # 점수만 추출
    average_score = sum(scores) / len(scores)        # 평균 계산
    print(f"선택된 리간드의 평균 점수: {average_score:.2f}")