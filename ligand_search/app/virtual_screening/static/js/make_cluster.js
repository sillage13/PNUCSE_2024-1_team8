$(document).ready(function () {
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

    $('.btn_div').hide()
    makeCluster();
    var resultLen = 0
    
    function makeCluster() {
        $.ajax({
            url: "/perform-cluster/",
            type: "POST",
            contentType: "application/json",
            success: function (data) {
                if (data.status === "started") {
                    checkStatus();
                }
            },
        });
    }
    
    function checkStatus() {
        $.ajax({
            url: "/get-task-status/",
            type: "GET",
            dataType: "json",
            success: function (data) {
                var resultsDiv = $("#results");
                var isScrollDown = resultsDiv[0].scrollHeight - resultsDiv.scrollTop() <  resultsDiv.outerHeight() + 1
                
                for (let i = resultLen; i < data.result.length; i++) {
                    var result = data.result.at(i)
                    //replace
                    result = result.replace("\n", "")
                    result = result.replace("<", "&lt;")
                    result = result.replace(">", "&gt;")
                    while (result.includes("  "))
                        result = result.replace("  ", "&emsp;")
                    
                    if (result.startsWith('??')) {
                        result = result.replace("??", "")
                        resultsDiv.children().last().remove()
                    }
                    resultsDiv.append("<span>" + result + "</span>")
                }
                resultLen = data.result.length
                //스크롤 이동
                if (isScrollDown)
                    resultsDiv.scrollTop(resultsDiv[0].scrollHeight)
                
                if (data.result.at(-1) != "Processing complete") {
                    setTimeout(checkStatus, 1000)
                }
                else {
                    let url = "/manage-ligand/"
                    $('.btn_div a').attr("href", url)
                    $('.btn_div').show()
                }
                // resultsDiv.html(data.result.replace("\r", "<br>"));
                // if (!data.result.includes("complete")) {
                //   setTimeout(checkStatus, 1000);
                // } else {
                //   // Optionally, redirect to the results page
                //   // window.location.href = "/results-list/";
                // }
            },
        });
    }
});
