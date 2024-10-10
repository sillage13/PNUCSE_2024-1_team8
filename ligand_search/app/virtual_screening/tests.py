from django.test import TestCase


# Create your tests here.

# from virtual_screening.models import Result, Ligand
# class ResultModelTest(TestCase):
#     def test_create_result(self):
#         result = Result.objects.create(
#             receptor_name='Test Receptor',
#             receptor_file='receptor_files/test_receptor.pdbqt',
#             result_directory='/screening/result/test_receptor_20230924_135530/',
#             ligand_1=Ligand.objects.create(ligand_smile='ligand1'), score_1=-10.0,
#             ligand_2=Ligand.objects.create(ligand_smile='ligand2'), score_2=-9.8,
#             ligand_3=Ligand.objects.create(ligand_smile='ligand3'), score_3=-9.5,
#             ligand_4=Ligand.objects.create(ligand_smile='ligand4'), score_4=-9.3,
#             ligand_5=Ligand.objects.create(ligand_smile='ligand5'), score_5=-9.0,
#             ligand_6=Ligand.objects.create(ligand_smile='ligand6'), score_6=-8.8,
#             ligand_7=Ligand.objects.create(ligand_smile='ligand7'), score_7=-8.5,
#             ligand_8=Ligand.objects.create(ligand_smile='ligand8'), score_8=-8.3,
#             ligand_9=Ligand.objects.create(ligand_smile='ligand9'), score_9=-8.0,
#             ligand_10=Ligand.objects.create(ligand_smile='ligand10'), score_10=-7.8,
#             search_method='Method XYZ',
#             execution_time=15.32,
#             status='successed'
#         )
        
#         self.assertEqual(result.receptor_name, 'Test Receptor')
#         self.assertEqual(result.status, 'successed')
#         self.assertEqual(result.execution_time, 15.32)

from virtual_screening.models import Ligand
class LigandModelTest(TestCase):
    def test_create_ligand(self):
        smiles = 