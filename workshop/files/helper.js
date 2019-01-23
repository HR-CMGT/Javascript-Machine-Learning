'use strict';


// ******************************************************************************************
//
// START THE WEBCAM STREAM
// docs: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
//
// ******************************************************************************************

function startWebcam() {
    if (navigator.mediaDevices) {
        navigator.mediaDevices.getUserMedia({ video: true })
            // permission granted:
            .then(function (stream) {
                video.srcObject = stream;
                video.addEventListener("playing", () => startApp())
            })
            // permission denied:
            .catch(function (error) {
                document.body.textContent = 'Could not access the camera. Error: ' + error.name
                alert("I can't let you do that, Dave")
            })
    }
}
// ******************************************************************************************
//
// color helper functions
// convert three R, G, B values into one single number that still holds R G B information
// this is not strictly needed, but your array will be three times as small!
//
// ******************************************************************************************
function rgbToDecimal(red, green, blue) {
    var r = red & 0xFF;
    var g = green & 0xFF;
    var b = blue & 0xFF;
    return (r << 24) + (g << 16) + (b << 8) + (1); // alpha = 1
}

function decimalToRgb(decNumber) {
    var red = decNumber >> 24 & 0xFF;
    var green = decNumber >> 16 & 0xFF;
    var blue = decNumber >> 8 & 0xFF;
    var alpha = decNumber & 0xFF;
    console.log("decimal number contains: " + red + "," + green + "," + blue)
}