import os
import pickle
from openbabel import pybel
from rdkit import Chem
from openbabel import openbabel
from vina import Vina
import numpy as np
import argparse
import pandas as pd
from rdkit.Chem import AllChem, DataStructs
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from collections import defaultdict
import random
import dask.dataframe as dd


def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

a = load_data("smile_list.dat")
cnt = 0

for i in a:
    cnt += 1
    print(i)
print(cnt)


# # 'all.txt' 파일을 읽어서 한 줄씩 출력하는 코드
# with open('all.txt', 'r') as file:
#     for line in file:
#         # rstrip()을 사용하여 각 줄의 끝에 있는 줄바꿈 문자를 제거합니다
#         print(line.rstrip())
#         cnt += 1

# print(cnt)
#2104318