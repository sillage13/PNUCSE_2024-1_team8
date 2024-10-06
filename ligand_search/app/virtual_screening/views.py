from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.urls import reverse
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .models import Ligand, Result
import subprocess
import threading
import shutil
import datetime
import os


# Create your views here.
taskOutputs = ''


def performTask(request):
    global taskOutputs
    taskOutputs = ''

    method = request.session.get('method')
    receptor = request.session.get('receptor')
    resultDir = request.session.get('resultDir')
    is_demo = request.session.get('is_demo')

    def run_script():
        global taskOutputs
        
        try:
            receptorPath = os.path.join(resultDir, receptor)
            count = '200000000'  # or any appropriate value
            cmd =[]

            scriptPath = ''
            if method == 'Random':
                scriptPath = '/screening/method/method_random.py'
                cmd = [
                    'python',
                    scriptPath,
                    receptorPath,
                    count,
                    is_demo,
                    resultDir
                ]
            elif method == 'Clustering':
                scriptPath = '/screening/method/method_clustering.py'
                cmd = [
                    'python',
                    scriptPath,
                    receptorPath,
                    count,
                    is_demo,
                    resultDir                    
                ]
            elif method == 'MEMES':
                scriptPath = '/screening/method/method_memes.py'
                
            

            

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
                taskOutputs += line
                print(line, end='')

            # process.stdout.close()
            process.wait()

            # After processing is complete, you might want to save results to the database
            # and redirect or notify the user

        except Exception as e:
            taskOutputs += f"\nError: {str(e)}"
            print(f"Error: {str(e)}")
        finally:
            taskOutputs += "\nProcessing complete"
            print("Processing complete")
            
            logFilePath = os.path.join(resultDir, "log.txt")
            
            with open(logFilePath, 'w') as f:
                f.write(taskOutputs)

    thread = threading.Thread(target=run_script)
    thread.start()

    return JsonResponse({'status': 'started'})


def getTaskStatus(request):
    global taskOutputs
    return JsonResponse({'result': taskOutputs})


def processing(request):
    method = request.session.get('method')
    receptor = request.session.get('receptor')
    resultDir = request.session.get('resultDir')
    is_demo = request.session.get('is_demo')

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

        return redirect(processing)

    return render(request, 'demo.html')


def search(request):
    if request.method == "POST":
        if not request.FILES:
            return render(request, 'search.html', {'receptor_error_message': 'Please upload receptor file'})
        
        method = request.POST.get('method')
        receptor_file = request.FILES["receptor"]
        receptor = receptor_file.name

        if not method:
            return render(request, 'search.html', {'method_error_message': 'Please select a method.'})

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

        return redirect(processing)
    return render(request, 'search.html')


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
        'ligand': result.ligand_1,
    })
