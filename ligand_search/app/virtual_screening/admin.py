from django.contrib import admin
from .models import Ligand
from .models import Result

# Register your models here.
admin.site.register(Ligand)
admin.site.register(Result)