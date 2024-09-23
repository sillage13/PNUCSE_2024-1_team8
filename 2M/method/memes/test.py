import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import time

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    # return joblib.load(filename, mmap_mode='r')
    
def save_data(target, filename):
    with open(filename, 'wb') as f:
        pickle.dump(target, f)
    # joblib.dump(target, filename)


# sl = load_data(f"./../../data/smile_list.dat")
# docking_dict = load_data(f"./../../data/smile_score_dict.dat")
# ligand_index = [f'ligand{str(i).zfill(7)}' for i in range(1, 2104319)]

# smile_index = dict()

# for i in range(2104318):
#     smile_index[sl[i]] = ligand_index[i]

# save_data(smile_index, "smile_index_dict.dat")


a = load_data("smile_index_dict.dat")
for i, v in a.items():
    print(i, v)
    time.sleep(0.3)

