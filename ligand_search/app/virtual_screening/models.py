from django.db import models
from django.conf import settings

# Create your models here.
class Ligand(models.Model):
    # 리간드 이름
    ligand_name = models.CharField(max_length=200, db_index=True)
    # 리간드 파일 생성 날짜
    created_at = models.DateTimeField(auto_now_add=True)
    # 리간드 파일 이름
    ligand_file_name = models.CharField(max_length=100)
    
    def __str__(self):
        return self.ligand_name
    
    @property
    def get_ligand_file_path(self):
        return f"{settings.LIGAND_FILE_PATH}{self.ligand_file_name}"