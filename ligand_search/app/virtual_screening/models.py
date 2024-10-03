from django.db import models

# Create your models here.
class Ligand(models.Model):
    # 리간드 스마일
    ligand_smile = models.TextField(unique=True)
    # 리간드 파일 생성 날짜
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.ligand_smile
    
    class Meta:
        indexes = [
            models.Index(fields=['id']),  # id 필드에 인덱스 추가
            models.Index(fields=['created_at']),  # created_at 필드에 인덱스 추가 (선택 사항)
        ]

    
class Result(models.Model):
    STATUS_CHOICES = [
        ('successed', 'Successed'),
        ('failed', 'Failed')
    ]

    receptor_name = models.CharField(max_length=255)
    receptor_file = models.FileField(upload_to='receptor_files/')  # Store pdbqt files
    result_directory = models.CharField(max_length=500)  # Directory path for log, output files, etc.

    # Top 10 Ligand Smiles and Scores
    ligand_1 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_1')
    score_1 = models.FloatField()
    ligand_2 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_2')
    score_2 = models.FloatField()
    ligand_3 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_3')
    score_3 = models.FloatField()
    ligand_4 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_4')
    score_4 = models.FloatField()
    ligand_5 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_5')
    score_5 = models.FloatField()
    ligand_6 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_6')
    score_6 = models.FloatField()
    ligand_7 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_7')
    score_7 = models.FloatField()
    ligand_8 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_8')
    score_8 = models.FloatField()
    ligand_9 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_9')
    score_9 = models.FloatField()
    ligand_10 = models.ForeignKey(Ligand, on_delete=models.SET_NULL, null=True, related_name='ligand_10')
    score_10 = models.FloatField()

    # Search Method and Execution Time
    search_method = models.CharField(max_length=100)
    execution_time = models.FloatField(help_text="Time taken for searching (in seconds)")

    # Status and Timestamps
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    started_at = models.DateTimeField(auto_now_add=True)
    ended_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Result for {self.receptor_name} ({self.status})"
    
    class Meta:
        indexes = [
            models.Index(fields=['receptor_name']),
            models.Index(fields=['status']),
        ]