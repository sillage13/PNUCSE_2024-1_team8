# your_app_name/management/commands/import_ligands.py

import pickle
import joblib
import numpy as np
from django.core.management.base import BaseCommand
from django.conf import settings
from virtual_screening.models import Ligand


class Command(BaseCommand):
    help = 'Imports ligand files into the database'

    def handle(self, *args, **kwargs):
        with open(settings.DATA_DIR / "demo/smile_list.dat", "rb") as file:
            smile_list = pickle.load(file)

        with open(settings.DATA_DIR / "demo/fingerprints.dat", "rb") as file:
            fingerprints = pickle.load(file)

        with open(settings.DATA_DIR / "demo/features.pkl", "rb") as file:
            features = np.nan_to_num(np.array(joblib.load(file, mmap_mode='r')))

        batch_size = 100000
        total = len(smile_list)

        # for i in range(0, total, batch_size):
        #     batch = [Ligand(ligand_smile=smile) for smile in smile_list[i:i + batch_size]]
        #     Ligand.objects.bulk_create(batch)

        for i in range(0, total, batch_size):
            batch = []
            smiles_batch = smile_list[i:i + batch_size]
            fingerprints_batch = fingerprints[i:i + batch_size]
            features_batch = features[i:i + batch_size]
            for smile, fingerprint, mol2vec in zip(smiles_batch, fingerprints_batch, features_batch):
                ligand = Ligand(
                    ligand_smile=smile,
                    fingerprint=fingerprint.tobytes(),
                    mol2vec=mol2vec.tobytes()
                )
                batch.append(ligand)
            Ligand.objects.bulk_create(batch)

        self.stdout.write(self.style.SUCCESS(
            'Successfully imported ligands from pickle'))
