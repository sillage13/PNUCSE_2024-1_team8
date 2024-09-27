$(document).ready(function() {
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