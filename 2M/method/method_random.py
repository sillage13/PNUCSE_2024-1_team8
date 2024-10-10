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

if __name__ == "__main__":

    start_time = time.time()
    
    parser = argparse.ArgumentParser(description="Docking script")
    parser.add_argument('--receptor', type=str, help='Path to the receptor file')
    parser.add_argument('--count', type=str, help='Docking count')
    parser.add_argument('--is_demo', type=str, help='Is demo')
    parser.add_argument('--result_dir', type=str, help='Result directory')
    args = parser.parse_args()
    receptor = args.receptor
    count = int(args.count)
    is_demo = args.is_demo
    result_dir = args.result_dir
    
    ligands = list()
    
    if is_demo == "True":
        ligands_score = load_data("/screening/data/demo/smile_score_dict.dat")
        ligands = list(ligands_score.keys())
    else:
        ligands = list(Ligand.objects.values_list('ligand_smile', flat=True))
    
    random_result = list()
    random_smiles = random.sample(ligands, count)
    
    for smile in random_smiles:
        try:
            if is_demo == "True":
                score = ligands_score[smile]
            else:
                score = cal_score(smile, receptor)
            
            random_result.append((score, smile))
        except Exception as e:
            print(f"Error processing {smile}: {e}")
            continue

    random_result.sort(key=lambda x: x[0])
    top_ligands = random_result[:10]

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
