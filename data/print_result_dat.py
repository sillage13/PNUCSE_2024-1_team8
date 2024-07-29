import os
import pickle
from openbabel import pybel
from rdkit import Chem
from openbabel import openbabel
from vina import Vina
import numpy as np
import argparse
import random


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Docking script")
    parser.add_argument('file', type=str, help='Path to the result file')
    args = parser.parse_args()
    file = args.file

    print(file)

    result = load_data(file)

    tmp = 0
    for i, v in result.items():
        print(i, v)
        tmp += v

    print(len(result))
    print(tmp / len(result))

