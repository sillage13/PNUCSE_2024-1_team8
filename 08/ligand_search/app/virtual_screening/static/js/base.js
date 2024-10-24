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

    $('#color_mode').click(function() {
        if (mode == 'dark')
            mode = 'light'
        else
            mode = 'dark'

        changeMode(mode)
    })
});