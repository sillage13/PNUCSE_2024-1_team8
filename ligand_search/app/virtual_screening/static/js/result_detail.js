$(document).ready(function() {
    function getCookie(name) {
        var cookieValue = null;
        if (document.cookie && document.cookie !== "") {
            var cookies = document.cookie.split(";");
            for (var i = 0; i < cookies.length; i++) {
                var cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === name + "=") {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
    var csrftoken = getCookie("csrftoken");
    
    function csrfSafeMethod(method) {
        return /^(GET|HEAD|OPTIONS|TRACE)$/.test(method);
    }
    $.ajaxSetup({
        beforeSend: function (xhr, settings) {
            if (!csrfSafeMethod(settings.type) && !this.crossDomain) {
                xhr.setRequestHeader("X-CSRFToken", csrftoken);
            }
        },
    });


    
    let element = document.querySelector('#visualization')    
    let root = document.querySelector(':root')
    let rootStyles = getComputedStyle(root)
    let backgroundColor = rootStyles.getPropertyValue('--background')

    let config = { 
        'id': 'visual_canvas',
        'backgroundColor': backgroundColor,
    };
    let viewer = $3Dmol.createViewer( element, config );

    $(".container > *:not(:last-child)").hide()

    jQuery.ajax({
        url: "/visualization",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({
            smiles: ligand_smile,
            id: ligand_id,
            receptor: receptor_name,
            result_dir: result_directory
        }),
        success: function(data) {
            $(".container > *").show()
            $(".sk-center").hide()
            visualization(data.receptor_path, data.ligand_path)
        },
        error: function(err) {
            console.error(err)
        }
    })
    
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

    $('#info_icon').click(function() {
        $('.visual-info:not(:first-child)').toggle()
        
        var txt = $(this).text()
        if (txt == "subtitles")
            $(this).text('subtitles_off')
        else
            $(this).text('subtitles')
    })

    $('#color_mode').click(function() {   
        backgroundColor = rootStyles.getPropertyValue('--background')
        viewer.setBackgroundColor(backgroundColor)
    })


    function visualization(receptorFile, ligandFile) {
        jQuery.ajax( receptorFile, { 
            success: function(data) {
                model = viewer.addModel( data, "pdb" );                       
                viewer.setStyle({"model": model}, {cartoon: {color: 'white'}});  
                viewer.zoomTo();                                     
                viewer.render();                                    
            },
            error: function(hdr, status, err) {
                console.error( "Failed to load Receptor " + receptorFile + ": " + err );
            },
        });
        jQuery.ajax( ligandFile, { 
            success: function(data) {
                model = viewer.addModel( data, "xyz" );                 
                viewer.setStyle({"model": model}, {stick: {colorscheme: 'greenCarbon'}});                                   
                viewer.render();                                    
            },
            error: function(hdr, status, err) {
                console.error( "Failed to load Ligand " + ligandFile + ": " + err );
            },
        });
    }
})