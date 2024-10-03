$(document).ready(function() {
    /*
    let element = document.querySelector('#visalization');
    let config = {  };
    let viewer = $3Dmol.createViewer( element, config );
    jQuery.ajax( receptorFile, { 
        success: function(data) {
            model = viewer.addModel( data, "pdb" );                       
            viewer.setStyle({"model": model}, {cartoon: {color: 'white'}});  
            viewer.zoomTo();                                     
            viewer.render();                                    
        },
        error: function(hdr, status, err) {
            console.error( "Failed to load PDB " + receptorFile + ": " + err );
        },
    });
    jQuery.ajax( ligandFile, { 
        success: function(data) {
            model = viewer.addModel( data, "xyz" );                 
            viewer.setStyle({"model": model}, {stick: {colorscheme: 'greenCarbon'}});                                   
            viewer.render();                                    
        },
        error: function(hdr, status, err) {
            console.error( "Failed to load PDB " + ligandFile + ": " + err );
        },
    });
    */

    $('.hidden-overflow_div').hover(
        function() {
            var txt = $(this).find(".hidden-overflow")
            if (txt[0].clientWidth < txt[0].scrollWidth) {
                txt.addClass("flow-text")
                $('.flow-text').css('animation-duration', txt[0].scrollWidth/30+'s')
            }
        },
        function() {
            $(this).find(".hidden-overflow").removeClass("flow-text")
        }
    )
})