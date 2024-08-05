$(document).ready(function() {
    function changeMode(mode) {
        localStorage.setItem('color-mode', mode)
        $(':root').attr('color-mode', mode)
        if (mode == 'dark')
            $('#color_mode').text('light_mode')
        else
            $('#color_mode').text('dark_mode')
    }

    //mode 설정
    let mode = localStorage.getItem('color-mode')
    if (!mode)
        mode = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    changeMode(mode)

    //창 너비에 따라 메뉴 아이콘 display 변경
    $(window).resize(function() {
        if (window.matchMedia('(max-width: 700px)').matches)
           $('.menu_item').css({'display': 'none'})
        else
            $('.menu_item').css({'display': 'block'})
    })

    //메뉴 아이콘 클릭시 메뉴 표시   
    $('#menu').click(function() {
        if (window.matchMedia('(max-width: 700px)').matches) {
            $('.menu_item').toggle()
            $('.menu_icon > .material-symbols-outlined').css({'top': $('.title').height()/2+10})
        }
    })

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

    //파일 선택 시 이름 표시
    $('#id_receptor, #id_ligand').change(function(event) {
        let filename = $(this).val()
        let id = event.target.id
        if (filename) {
            $('label[for='+id+']').text(filename)
            $('label[for='+id+']').css({'color':'var(--on-container)'}) 
        }
        
    })

    $('#color_mode').click(function() {
        if (mode == 'dark')
            mode = 'light'
        else
            mode = 'dark'

        changeMode(mode)
    })
});