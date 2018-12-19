var t1 = 320;
var d1 = 140;

$(window).scroll(function() {
    var scroll = $(document).scrollTop();
    console.log(scroll); 
    if (scroll < t1) {
    	$("#header").css({'opacity': newOpac});
    } else if (scroll >= t1 && scroll <= t1 + d1) {
        var newOpac = .75 + (((scroll - t1) / d1) * .25);
        $("#header").css({'opacity': newOpac});
    } else if (scroll >= t1 + d1) {
        $("#header").css({'opacity': 1});
    } else {
    	$("#header").css({'opacity': 0.75}); 
    }
});
