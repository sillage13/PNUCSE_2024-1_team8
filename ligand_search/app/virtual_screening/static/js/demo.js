// demo.js

$(document).ready(function() {
    $('#method').val(null)
    
    // Implement dropdown menu
    $('label[for=method]').click(function() {
        $('.select_ul').toggle()
        $('#arrow_drop_down').toggleClass('flip')
    })

    $('.select_li').click(function() {
        let method = $(this).text()
        $('#method').val(method)
        $('label[for=method]').text(method)
        $('label[for=method]').css({'color':'var(--on-container)'})
        $('label[for=method]').click()
    })
})
