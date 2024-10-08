$(document).ready(function() {
    if (window.matchMedia('(hover:hover) and (pointer:fine').matches) {
        $('.result_receptor, .result_method').hover(
            function() {
                if ($(this).hasClass('result_receptor')) 
                    var sibling = '.result_method'
                else
                    var sibling = '.result_receptor'

                var txt = $(this).find(".hidden-overflow")
                if (txt[0].clientWidth < txt[0].scrollWidth) {
                    $(this).addClass('show-text')
                    $(this).siblings(sibling).hide()

                    if (txt[0].clientWidth < txt[0].scrollWidth) {
                        txt.addClass("flow-text")
                        $('.flow-text').css('animation-duration', txt[0].scrollWidth/30+'s')
                    }
                }
            },
            function() {
                if ($(this).hasClass('result_receptor'))
                    var sibling = '.result_method'
                else
                    var sibling = '.result_receptor'

                $(this).removeClass('show-text')
                $(this).siblings(sibling).show()

                $(this).find(".hidden-overflow").removeClass("flow-text")
            }
        )
    }
    else {
        $('.result_receptor, .result_method').each(function() {
            var txt = $(this).find(".hidden-overflow")
            if (txt[0].clientWidth < txt[0].scrollWidth) {
                txt.addClass("flow-text")
                $('.flow-text').css('animation-duration', txt[0].scrollWidth/30+'s')
            }
        })
    }
})