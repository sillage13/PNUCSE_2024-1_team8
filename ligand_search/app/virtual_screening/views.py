from django.shortcuts import render

# Create your views here.
def search(request):
    return render(request, 'search.html')

def manageLigand(request):
    return render(request, "manage_ligand.html")