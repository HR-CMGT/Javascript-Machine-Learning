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
// grayscale helper function
// convert three R, G, B values with range 0-255 into one single number with range 0-1
//
// ******************************************************************************************
function rgbToGrayscale(red, green, blue) {
    // convert red, green, blue values to one number (brightness)
    let grayscale = 0.30 * red + 0.59 * green + 0.11 * blue
    // number should range from 0 to 1 
    return grayscale / 255
}
// ******************************************************************************************
//
// draw your network as an svg. 
//
// ******************************************************************************************
/* 
    // example code to draw svg
    const config = {
        inputSize: 3,
        inputRange: 3,
        hiddenLayers: [4],
        outputSize: 3
    };
    const net = new brain.NeuralNetwork(config)
    let options = getSvgOptions()
    document.getElementById('result').innerHTML = brain.utilities.toSVG(net, options)
*/
function getSvgOptions() {
    return {
        fontSize: "12px",
        width: 600,
        height: 400,
        radius: 6,
        line: { width: 0.5, color: "rgba(0,0,0,1)" },
        inputs: { color: "rgba(0,127,0,0.6)", label: [] },
        hidden: { color: "rgba(255,127,80,0.6)" },
        outputs: { color: "rgba(100,149,237,0.6)" }
    }
}