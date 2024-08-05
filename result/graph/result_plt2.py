import matplotlib.pyplot as plt
import numpy as np
import pickle

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# 데이터 로드
a = load_data("./../4EK3/120/scoring_result_20.dat")
b = load_data("./../4EK3/120/scoring_result_20.dat")
c = load_data("./../4EK3/120/scoring_result_20.dat")

# 각 리스트에서 값만 추출
sample1_scores = [ligand[0] for ligand in a]
sample2_scores = [ligand[0] for ligand in b]
sample3_scores = [ligand[0] for ligand in c]

# 평균값 계산
avg_sample1_score = sum(sample1_scores) / len(sample1_scores)
avg_sample2_score = sum(sample2_scores) / len(sample2_scores)
avg_sample3_score = sum(sample3_scores) / len(sample3_scores)

# 상위 10개 추출
top_sample1 = a[:10]
top_sample2 = b[:10]
top_sample3 = c[:10]

# 상위 10개의 값만 추출
top_sample1_score = [ligand[0] for ligand in top_sample1]
top_sample2_score = [ligand[0] for ligand in top_sample2]
top_sample3_score = [ligand[0] for ligand in top_sample3]

# 상위 10개의 평균값 계산
avg_top_score1 = sum(top_sample1_score) / len(top_sample1_score)
avg_top_score2 = sum(top_sample2_score) / len(top_sample2_score)
avg_top_score3 = sum(top_sample3_score) / len(top_sample3_score)

# 전체 평균값의 평균 계산
total_avg_sample_score = (avg_sample1_score + avg_sample2_score + avg_sample3_score) / 3
total_avg_top_score = (avg_top_score1 + avg_top_score2 + avg_top_score3) / 3

# 상위 10개의 모든 리간드 점수와 인덱스 생성
all_top_scores = top_sample1_score + top_sample2_score + top_sample3_score
all_top_indices = [4] * len(all_top_scores)

# 인덱스 생성
indices1 = [1] * len(top_sample1_score)
indices2 = [2] * len(top_sample2_score)
indices3 = [3] * len(top_sample3_score)

# 색상 설정
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# 플롯 생성
plt.figure(figsize=(10, 6))

# 각 리스트 값 플로팅
plt.scatter(top_sample1_score, indices1, color=colors[0], marker='o', label='Top 10 ligands')
# plt.scatter(top_sample2_score, indices2, color=colors[0], marker='o')
# plt.scatter(top_sample3_score, indices3, color=colors[0], marker='o')

# 평균값 플로팅
plt.scatter([avg_sample1_score], [1], color=colors[1], marker='s', s=100, label='Sample average')
# plt.scatter([avg_sample2_score], [2], color=colors[1], marker='s', s=100)
# plt.scatter([avg_sample3_score], [3], color=colors[1], marker='s', s=100)

# 상위 10개의 평균값 플로팅
plt.scatter([avg_top_score1], [1], color=colors[2], marker='^', s=100, label='Top 10 average')
# plt.scatter([avg_top_score2], [2], color=colors[2], marker='^', s=100)
# plt.scatter([avg_top_score3], [3], color=colors[2], marker='^', s=100)

# 상위 10개의 모든 리간드 점수 플로팅
# plt.scatter(all_top_scores, all_top_indices, color=colors[3], marker='o', label='Total top 10 Ligands')

# 전체 평균값의 평균 플로팅
# plt.scatter([total_avg_sample_score], [4], color=colors[5], marker='s', s=100, label='Average of sample average')
# plt.scatter([total_avg_top_score], [4], color=colors[4], marker='^', s=100, label='Average of top 10 average')

# x축 반전 설정

# 그래프 제목과 라벨 설정
plt.title('Scoring')
plt.xlabel('Scores')
plt.yticks([1], ['Case 1'])
plt.legend()
plt.xlim(-11.5, -6)
plt.gca().invert_xaxis()
plt.axvline(x=avg_top_score1, color='red', linestyle='--', linewidth=2, label='Top 10 average line')
plt.annotate(f'{avg_top_score1:.2f}', 
             xy=(avg_top_score1, plt.gca().get_ylim()[0]),
             xytext=(0, -7), # y축 아래로 25 포인트
             textcoords='offset points',
             ha='center', va='top',
             color='red', fontweight='bold')

plt.grid(True)

# 그래프 저장
plt.savefig('scoring_result_120.png')
