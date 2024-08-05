import numpy as np
import random
import os
import pickle
from collections import defaultdict
import argparse

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main(filename):
    sample = load_data(filename)
    top_ligands = sample[:10]
    # print(sample)
    
    print("Top10 ligands")
    for ligand in top_ligands:
        print(f"File: {ligand[1]}, Score: {ligand[0]}")
    print()
    top_scores = [ligand[0] for ligand in top_ligands]
    sample_scores = [ligand[0] for ligand in sample]
    avg_top_score = sum(top_scores) / len(top_scores)
    avg_sample_score = sum(sample_scores) / len(sample_scores)
    print(f"Average score of top10 ligands: {avg_top_score:.2f}")
    print(f"Average score of sample ligands: {avg_sample_score:.2f}")
    print(f"len: {len(sample)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load and process ligand data.')
    parser.add_argument('filename', type=str, help='The filename of the data file to load.')
    
    args = parser.parse_args()
    main(args.filename)
