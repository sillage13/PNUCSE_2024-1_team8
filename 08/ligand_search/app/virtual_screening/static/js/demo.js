// demo.js

$(document).ready(function() {
    $('#method').val(null)
    $('.sub-input').hide()
    
    // Implement dropdown menu
    $('label[for=method]').click(function() {
        $(this).siblings('.select_ul').toggle()
        $(this).siblings('#arrow_drop_down').toggleClass('flip')
    })

    $('#method_select > .select_li').click(function() {
        let method = $(this).text()
        $('#method').val(method)
        $('label[for=method]').text(method)
        $('label[for=method]').css({'color':'var(--on-container)'})
        $('label[for=method]').click()
        if (method=="MEMES")
            $('.sub-input').show()
        else
            $('.sub-input').hide()
    })

    $('label[for=af]').click(function() {
        $(this).siblings('.select_ul').toggle()
        $(this).siblings('#arrow_drop_down').toggleClass('flip')
    })

    $('#af_select > .select_li').click(function() {
        let af = $(this).text()
        $('#af').val(af)
        $('label[for=af]').text(af)
        $('label[for=af]').css({'color':'var(--on-container)'})
        $('label[for=af]').click()
    })
})
