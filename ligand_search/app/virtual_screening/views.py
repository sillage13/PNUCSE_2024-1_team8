from django.shortcuts import render, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.conf import settings
from django.db import transaction
from .models import Ligand
from .forms import LigandForm
from .utils import is_valid_pdbqt, get_ligand_name, get_unique_file_name
import os
    
# Create your views here.
def search(request):
    if request.method =="POST":
        return HttpResponseRedirect(reverse('result'))

    return render(request, 'search.html')

def manageLigand(request):
    error_message = None
    
    if request.method == 'POST':
        form = LigandForm(request.POST, request.FILES)
        
        # 폼이 유효하지 않은 경우 (파일 선택 X)
        if not form.is_valid():
            error_message = 'Please select file.'
            return render(request, 'manage_ligand.html', {'error_message': error_message, 'ligands': Ligand.objects.all()})
            
        ligand_file = form.cleaned_data['ligand_file']
        
        # pdbqt 파일이 맞는지, 올바른지 확인
        if not is_valid_pdbqt(ligand_file) or not ligand_file.name.endswith('.pdbqt'):
            error_message = 'Invalid file: File must be in pdbqt format'
            return render(request, 'manage_ligand.html', {'error_message': error_message, 'ligands': Ligand.objects.all()})
    
        # pdbqt 파일에서 리간드 이름 추출
        ligand_name = get_ligand_name(ligand_file)
        
        # 리간드 이름 추출 실패
        if not ligand_name:
            error_message = 'Failed to extract ligand name: File must contain ligand name'
            return render(request, 'manage_ligand.html', {'error_message': error_message, 'ligands': Ligand.objects.all()})
        
        """
        # 이미 존재하는 리간드
        if Ligand.objects.filter(ligand_name=ligand_name).exists():
            error_message = 'Ligand with the same name already exists'
            return render(request, 'manage_ligand.html', {'error_message': error_message, 'ligands': Ligand.objects.all()})
        """
        
        # 리간드 저장 시 파일 이름 유니크화
        file_name = get_unique_file_name(ligand_file.name)
        file_path = os.path.join(settings.LIGAND_FILE_PATH, file_name)
        
        with transaction.atomic():
            # 데이터베이스에 임시 저장
            ligand = Ligand(ligand_name=ligand_name, ligand_file_name=file_name)
            ligand.save()
            
            try:
                # pdbqt 파일 저장
                with open(file_path, 'wb') as f:
                    for chunk in ligand_file.chunks():
                        f.write(chunk)
            except Exception as e:
                # 파일 저장 실패 시 롤백
                form.add_error('ligand_file', 'Failed to save ligand file')
                ligand.delete()
                
                raise e
        return redirect('manage-ligand')
    
    return render(request, 'manage_ligand.html', {'error_message': error_message, 'ligands': Ligand.objects.all()})

def result(request):
    return render(request, 'result.html')