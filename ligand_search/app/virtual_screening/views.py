from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.core.files.storage import FileSystemStorage
from django.core.management import call_command
from django.conf import settings
from django.contrib import messages
from .models import Ligand, Result
import subprocess
import threading
import shutil
import datetime
import os
import json


# Create your views here.
taskOutputs = []


def performTask(request):
    global taskOutputs
    taskOutputs.clear()

    method = request.session.get('method')
    receptor = request.session.get('receptor')
    resultDir = request.session.get('resultDir')
    is_demo = request.session.get('is_demo')
    
    receptorPath = os.path.join(resultDir, receptor)
    
    from django.utils import timezone
    result = Result(
        receptor_name=receptor,
        receptor_file=receptorPath,
        result_directory=resultDir,
        search_method=method,
        execution_time=0,
        status='failed',
        started_at=timezone.now(),        
    )
    
    result.save()
    request.session['result_id'] = result.id

    def run_script():
        global taskOutputs
        
        try:
            count = '20000'  # or any appropriate value
            cmd =[]

            scriptPath = ''
            if method == 'Random':
                scriptPath = '/screening/method/method_random.py'
                cmd = [
                    'python',
                    scriptPath,
                    '--receptor', receptorPath,
                    '--count', count,
                    '--is_demo', is_demo,
                    '--result_dir', resultDir
                ]
            elif method == 'Clustering':
                scriptPath = '/screening/method/method_clustering.py'
                cmd = [
                    'python',
                    scriptPath,
                    '--receptor', receptorPath,
                    '--count', count,
                    '--is_demo', is_demo,
                    '--result_dir', resultDir                   
                ]
            elif method == 'MEMES':
                scriptPath = '/screening/method/method_memes.py'
                # todo
                feature = request.session.get('representation')
                acquisition_func = request.session.get('af')
                featuresPath = '/screening/data/demo/features.pkl' if feature == "Mol2vec" else '/screening/data/demo/fingerprints.dat'
                cmd = [
                    'python',
                    scriptPath,
                    '--run', '1',
                    '--rec', 'memes_data',
                    '--cuda', 'cpu',
                    '--feature', feature,
                    '--features_path', featuresPath,
                    '--iter', '6',
                    '--capital', '30000',
                    '--initial', '8000',
                    '--periter', '2000',
                    '--n_cluster', '4000',
                    '--acquisition_func', acquisition_func,
                    '--receptor', receptorPath,
                    '--total_count', count,
                    '--is_demo', is_demo,
                    '--result_dir', resultDir
                ]
                

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Read the output line by line
            for line in process.stdout:
                # Append the line to the task output
                taskOutputs.append(line)

            # process.stdout.close()
            process.wait()
            
            resultJsonPath = os.path.join(resultDir, "result.json")
            if os.path.exists(resultJsonPath):
                with open(resultJsonPath, 'r') as f:
                    resultData = json.load(f)

                # Update the Result object
                result = Result.objects.get(id=request.session.get('result_id'))
                top_ligands = resultData['top_ligands']
                execution_time = resultData.get('execution_time', 0)
                result.execution_time = execution_time
                result.status = 'successed'
                result.ended_at = timezone.now()

                # Create or get Ligand objects and set them in the Result object
                for i, item in enumerate(top_ligands):
                    ligand_smile = item['smile']
                    score = item['score']
                    ligand_obj, _ = Ligand.objects.get_or_create(ligand_smile=ligand_smile)
                    setattr(result, f'ligand_{i+1}', ligand_obj)
                    setattr(result, f'score_{i+1}', score)
                    if i >= 9:
                        break  # Only up to ligand_10

                result.save()

        except Exception as e:
            taskOutputs.append(f"Error: {str(e)}")
        finally:
            taskOutputs.append("Processing complete")
            print("complete")
            
            logFilePath = os.path.join(resultDir, "log.txt")
            
            with open(logFilePath, 'w') as f:
                for line in taskOutputs:
                    f.write(line)

    thread = threading.Thread(target=run_script)
    thread.start()

    return JsonResponse({'status': 'started', 'result_id': request.session.get('result_id')})


def getTaskStatus(request):
    global taskOutputs
    return JsonResponse({'result': taskOutputs, 'result_id': request.session.get('result_id')})


def processing(request):
    method = request.session.get('method')
    receptor = request.session.get('receptor')
    resultDir = request.session.get('resultDir')

    if not method or not receptor or not resultDir:
        return redirect('demo')

    return render(request, 'processing.html', {
        'method': method,
        'receptor': receptor,
    })


def demo(request):
    if request.method == "POST":
        method = request.POST.get('method')
        receptor = request.POST.get('receptor')

        if not method:
            return render(request, 'demo.html', {'method_error_message': 'Please select a method.'})
        elif method == 'MEMES':
            representation = request.POST.get('representation')
            af = request.POST.get('af')
            if not representation or not af:
                return render(request, 'demo.html', {'method_error_message': 'Please select representation and acquisition function.'})

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        resultDirName = f"{receptor}_{timestamp}_{method}"
        resultDir = os.path.join(settings.RESULT_DIR, resultDirName)
        os.makedirs(resultDir, exist_ok=True)

        # 기존 리셉터 파일을 결과 폴더로 복사
        receptor_source = os.path.join(settings.DATA_DIR, 'receptor', receptor)
        receptor_dest = os.path.join(resultDir, receptor)
        shutil.copyfile(receptor_source, receptor_dest)

        request.session['method'] = method
        request.session['receptor'] = receptor
        request.session['resultDir'] = resultDir
        request.session['is_demo'] = "True"
        if method == 'MEMES':
            request.session['representation'] = representation
            request.session['af'] = af

        return redirect(processing)

    return render(request, 'demo.html')


def search(request):
    if request.method == "POST":
        try:    
            receptor_file = request.FILES["receptor"]
            receptor = receptor_file.name   
        except:
            return render(request, 'search.html', {'receptor_error_message': 'Please select a receptor file.'})
        
        method = request.POST.get('method')
        if not method:
            return render(request, 'search.html', {'method_error_message': 'Please select a method.'})
        elif method == 'MEMES':
            representation = request.POST.get('representation')
            af = request.POST.get('af')
            if not representation or not af:
                return render(request, 'search.html', {'method_error_message': 'Please select representation and acquisition function.'})

        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        resultDirName = f"{receptor}_{timestamp}_{method}"
        resultDir = os.path.join(settings.RESULT_DIR, resultDirName)
        os.makedirs(resultDir, exist_ok=True)

        fs = FileSystemStorage(location=resultDir)
        fs.save(receptor, receptor_file)

        request.session['method'] = method
        request.session['receptor'] = receptor
        request.session['resultDir'] = resultDir
        request.session['is_demo'] = "False"
        if method == 'MEMES':
            request.session['representation'] = representation
            request.session['af'] = af

        return redirect(processing)
    return render(request, 'search.html')


def import_from_db(request):
    if request.method == 'POST':
        try:
            call_command('import_ligands')
            messages.success(request, 'Successfully imported ligands from pre-made database.')
        except Exception as e:
            messages.error(request, f'Error importing ligands: {str(e)}')
        
        return redirect('manage-ligand')
    
    return redirect('manage-ligand')


def manageLigand(request):
    error_message = None

    # 페이지 오브젝트 생성
    paginator = Paginator(Ligand.objects.only('id').order_by('id'), 10)
    try:
        page_num = request.GET.get("page")
        page_obj = paginator.page(page_num)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    if request.method == 'POST':
        ligand_smile = request.POST.get('ligand_smile', '').strip()

        if not ligand_smile:
            error_message = "Please enter ligand information."
            return render(request, 'manage_ligand.html', {'error_message': error_message, 'page_obj': page_obj})
        
        #TODO 올바른 리간드 체크

        if Ligand.objects.filter(ligand_smile=ligand_smile).exists():
            error_message = 'Ligand already exists.'
            return render(request, 'manage_ligand.html', {'error_message': error_message, 'page_obj': page_obj})

        ligand = Ligand(ligand_smile=ligand_smile)
        ligand.save()

        return redirect('manage-ligand')

    return render(request, 'manage_ligand.html', {'error_message': error_message, 'page_obj': page_obj})


def results_list(request):
    paginator = Paginator(Result.objects.order_by('-id'), 10)

    try:
        page_num = request.GET.get("page")
        page_obj = paginator.page(page_num)
    except PageNotAnInteger:
        page_obj = paginator.page(1)
    except EmptyPage:
        page_obj = paginator.page(paginator.num_pages)

    return render(request, 'results_list.html', {'page_obj': page_obj})


def result_detail(request, result_id):
    result = get_object_or_404(Result, id=result_id)
    if result.status == "failed":
        return redirect("results-list")

    ligand_id = request.GET.get("ligand", result.ligand_1.id)
    ligand = get_object_or_404(Ligand, id=ligand_id)

    # receptor_path = result.result_directory
    # ligand_path = result.result_directory
    receptor_path = "/screening/result/test/4UNN.pdb"
    ligand_path = "/screening/result/test/output.xyz"

    ligands_scores = [
        {'ligand': result.ligand_1, 'score': result.score_1},
        {'ligand': result.ligand_2, 'score': result.score_2},
        {'ligand': result.ligand_3, 'score': result.score_3},
        {'ligand': result.ligand_4, 'score': result.score_4},
        {'ligand': result.ligand_5, 'score': result.score_5},
        {'ligand': result.ligand_6, 'score': result.score_6},
        {'ligand': result.ligand_7, 'score': result.score_7},
        {'ligand': result.ligand_8, 'score': result.score_8},
        {'ligand': result.ligand_9, 'score': result.score_9},
        {'ligand': result.ligand_10, 'score': result.score_10},
    ]

    return render(request, 'result_detail.html', {
        'result': result,
        'ligands_scores': ligands_scores,
        'ligand': ligand,
        'receptor_path': receptor_path,
        'ligand_path': ligand_path,
    })
