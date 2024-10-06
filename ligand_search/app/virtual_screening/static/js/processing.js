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

  performTask();

  function performTask() {
    var receptor = $("#results").data("receptor");
    var method = $("#results").data("method");

    $.ajax({
      url: "/perform-task/",
      type: "POST",
      contentType: "application/json",
      data: JSON.stringify({
        receptor: receptor,
        method: method,
      }),
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
        resultsDiv.html(data.result.replace("\r", "<br>"));
        if (!data.result.includes("complete")) {
          setTimeout(checkStatus, 1000);
        } else {
          // Optionally, redirect to the results page
          // window.location.href = "/results-list/";
        }
      },
    });
  }
});
