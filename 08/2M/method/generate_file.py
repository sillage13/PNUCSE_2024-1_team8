import os
import sys
import django
import argparse
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from utils import save_data, load_data

# Set up Django environment
sys.path.append("/app")  # Adjust to your Django project path
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "virtual_screening.settings")
django.setup()

from virtual_screening.models import Ligand

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_demo', type=str, help='Is demo')
    parser.add_argument('--feature', type=str)
    args = parser.parse_args()
    is_demo = (args.is_demo == "True")
    feature = args.feature

    feature_list = list()
    if is_demo:
        dirPath = "/screening/data/demo/"
        all_smiles = load_data(dirPath + "smile_list.dat")
    else:
        dirPath = "/screening/data/"
        all_smiles = list(Ligand.objects.values_list('ligand_smile', flat=True))

    if feature == 'Mol2vec':
        model = word2vec.Word2Vec.load("/screening/data/model_300dim.pkl")

        for i, smiles in enumerate(all_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.MolFromSmiles(smiles)
                sentence = MolSentence(mol2alt_sentence(mol, 1))
                del mol

                vec = sentences2vec([sentence], model, unseen="UNK")
                del sentence
                vec = DfVec(vec[0]).vec

                feature_list.append(vec)
        save_data(feature_list, dirPath+"features_mol2vec.pkl")
    else:
        for i, smiles in enumerate(all_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.MolFromSmiles(smiles)
                fingerprint = Chem.RDKFingerprint(mol, fpSize=1024)
                del mol

                feature_list.append(fingerprint)
        save_data(feature_list, dirPath+"features_fp.pkl")
