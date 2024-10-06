$(document).ready(function() {
    //파일 선택 시 이름 표시
    $('#receptor').change(function(event) {
        let filename = $(this).val()
        let id = event.target.id
        if (filename) {
            $('label[for='+id+']').text(filename)
            $('label[for='+id+']').css({'color':'var(--on-container)'}) 
        }
    })
    
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