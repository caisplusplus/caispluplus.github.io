var t1 = 464;
var d1 = 146;
var t2 = 4468;
var d2 = 146;

$(window).scroll(function() {
    var scroll = $(document).scrollTop();
    if (scroll > t1 && scroll < t1 + d1) {
        var newOpac = .75 + (((scroll - t1) / d1) * .25);
        $("#header").css({'opacity': newOpac});
    } else if (scroll > t1 + d1) {
        $("#header").css({'opacity': 1});
    } else if (scroll > t2) {
        var newOpac = 1 - (((scroll - t2) / d2) * .25);
        $("#header").css({'opacity': newOpac});
    } else {
        $("#header").css({'opacity': 0.75});
    }
});
