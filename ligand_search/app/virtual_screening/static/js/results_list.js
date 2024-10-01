$(document).ready(function() {
    $('.result_receptor, .result_method').hover(
        function() {
            if ($(this).hasClass('result_receptor')) 
                var sibling = '.result_method'
            else
                var sibling = '.result_receptor'

            var txt = $(this).find(".hidden-overflow")
            console.log(txt[0].clientWidth)
            console.log(txt[0].scrollWidth)
            if (txt[0].clientWidth < txt[0].scrollWidth) {
                $(this).addClass('show-text')
                $(this).siblings(sibling).hide()

                if (txt[0].clientWidth < txt[0].scrollWidth) {
                    txt.addClass("flow-text")
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
})