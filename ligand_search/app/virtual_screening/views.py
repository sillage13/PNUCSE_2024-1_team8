from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse

# Create your views here.
def search(request):
    if request.method =="POST":
        return HttpResponseRedirect(reverse('result'))

    return render(request, 'search.html')

def manageLigand(request):
    return render(request, 'manage_ligand.html')

def result(request):
    return render(request, 'result.html')