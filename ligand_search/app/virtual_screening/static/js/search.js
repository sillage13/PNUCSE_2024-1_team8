$(document).ready(function() {
    //폼에 element 추가
    $('.form_ul > li:first-child').prepend('<label class="label">Receptor</label>')
    $('.form_ul > li:first-child').append('<span class="material-symbols-outlined">upload</span>')

    $('#method').val(null)
    
    //드롭 다운 메뉴 구현
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