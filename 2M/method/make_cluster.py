import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import sys
import os
import django

# Set up Django environment
sys.path.append("/app")  # Adjust to your Django project path
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "virtual_screening.settings")
django.setup()

from virtual_screening.models import Ligand


def cluster_fingerprints(fingerprints, n_clusters, batch_size=10000, random_state=42):
    n_samples, n_features = fingerprints.shape
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, batch_size=batch_size
    )

    # 데이터 셔플
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    # 초기 피팅
    initial_batch = fingerprints[indices[:batch_size]]
    kmeans.partial_fit(initial_batch)

    # 나머지 데이터에 대해 부분 피팅
    for start in tqdm(range(batch_size, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch = fingerprints[indices[start:end]]
        kmeans.partial_fit(batch)

    # 최종 예측
    labels = np.zeros(n_samples, dtype=int)
    for start in tqdm(range(0, n_samples, batch_size)):
        end = min(start + batch_size, n_samples)
        batch = fingerprints[start:end]
        labels[start:end] = kmeans.predict(batch)

    return labels

# 클러스터 수
k_values_array = [500, 1000, 2000, 4000, 8000]

# 데이터 로드
fingerprints = []

print("Loading fingerprints... Please wait.")

fingerprints_bytes_array = Ligand.objects.values_list("fingerprint", flat=True)
dtype = np.int8

for fingerprints_bytes in fingerprints_bytes_array:
    fingerprints.append(np.frombuffer(fingerprints_bytes, dtype=dtype))
    
fingerprints = np.array(fingerprints)

# 클러스터링 수행
for k_values in k_values_array:
    labels = cluster_fingerprints(fingerprints, n_clusters=k_values)

    # 결과 저장
    output_file = f"/screening/data/labels{k_values}.txt"
    np.savetxt(output_file, labels, fmt="%d")

    print(f"Clustering results saved to {output_file}")