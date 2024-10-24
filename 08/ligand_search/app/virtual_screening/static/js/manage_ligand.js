$(document).ready(function() {
    $('#id_ligand').on("input", function() {
        let smiles = $(this).val()
        if (smiles) {
            $(this).css({'color':'var(--on-container)'})
        }
    })
})