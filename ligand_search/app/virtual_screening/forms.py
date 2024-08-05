from django import forms

class LigandForm(forms.Form):
    ligand_file = forms.FileField(label='ligand_file')