# your_app_name/management/commands/import_ligands.py

import os
import re
from django.core.management.base import BaseCommand
from django.conf import settings
from virtual_screening.models import Ligand
from django.db import transaction

class Command(BaseCommand):
    help = 'Imports ligand files into the database'

    def handle(self, *args, **kwargs):
        ligand_dir = settings.LIGAND_FILE_PATH

        ligand_name_re = re.compile(r'^REMARK\s+Name\s*=\s*(\S+)', re.MULTILINE)

        def extract_ligand_name(file_content):
            match = ligand_name_re.search(file_content)
            if match:
                return match.group(1)
            return None

        for file_name in os.listdir(ligand_dir):
            if file_name.endswith('.pdbqt'):
                file_path = os.path.join(ligand_dir, file_name)
                with open(file_path, 'r') as file:
                    file_content = file.read()
                    ligand_name = extract_ligand_name(file_content)
                    self.stdout.write(f'Processing {file_name}...')
                    if ligand_name:
                        with transaction.atomic():
                            ligand = Ligand(ligand_name=ligand_name, ligand_file_name=file_name)
                            ligand.save()
                    
            

        self.stdout.write(self.style.SUCCESS("Finished processing ligand files."))
