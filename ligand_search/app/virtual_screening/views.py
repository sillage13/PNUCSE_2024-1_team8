from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage
from .models import Ligand, Result
import os

# Create your views here.
def demo(request):
    return render(request, 'demo.html')


def search(request):
    if request.method == "POST":
        return HttpResponseRedirect(reverse('results-list'))

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
