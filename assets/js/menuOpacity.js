var t1 = 480;
var d1 = 146;
var t2 = 10000;
var d2 = 10000;

$(window).scroll(function() {
    var scroll = $(document).scrollTop();
    console.log("Scroll: " + scroll);
    if (scroll > t1 && scroll < t1 + d1) {
        var newOpac = .75 + (((scroll - t1) / d1) * .25);
        $("#header").css({'opacity': newOpac});
        console.log("Opacity: " + newOpac);
    } else if (scroll > t1 + d1 && scroll < t2) {
        $("#header").css({'opacity': 1});
    } else if (scroll > t2) {
        var newOpac = 1 - (((scroll - t2) / d2) * .25);
        $("#header").css({'opacity': newOpac});
        console.log('Opacity: ' + newOpac);
    } else {
        $("#header").css({'opacity': 0.75});
    }
});
