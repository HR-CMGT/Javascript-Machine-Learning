'use strict';

const video = document.querySelector('video')
const canvas = document.querySelector('#mosaic')
const context = canvas.getContext('2d')

// tiny canvas, to read pixel data from
const tiny_canvas = document.querySelector('#tiny_canvas')
const tiny_context = tiny_canvas.getContext('2d')

let width;// 340 
let height;// 255
let intervalid;

// how many pixels do we want to sample in each row and column?
let numpixels = 10

// ******************************************************************************************
//
// start the webcam stream. function is in helper.js
//
// ******************************************************************************************
startWebcam()


// ******************************************************************************************
//
// startapp is called when the webcam successfully runs
//
// ******************************************************************************************
function startApp() {
    let trainbuttons = document.getElementsByClassName("train")
    for(let b of trainbuttons){
        b.addEventListener("click", (e) => trainNetwork(e))
    }

    let startbutton = document.getElementsByClassName("start")
    startbutton.addEventListener("click", (e) => startClassification(e))

    width = video.offsetWidth
    height = video.offsetHeight

    canvas.width = width
    canvas.height = height

    tiny_canvas.width = width
    tiny_canvas.height = height

    context.mozImageSmoothingEnabled = false
    context.webkitImageSmoothingEnabled = false
    context.imageSmoothingEnabled = false

    // start drawing the webcam stream into the canvas so we can sample the colors
    drawWebcam()
}

// ******************************************************************************************
//
// convert the webcam stream to a canvas image
//
// ******************************************************************************************
function drawWebcam() {

    // Drawing the video very small and then blow it up to see what is happening
    context.drawImage(video, 0, 0, numpixels, numpixels)
    context.drawImage(canvas, 0, 0, numpixels, numpixels, 0, 0, width, height)

    // just draw a super tiny canvas
    tiny_context.drawImage(video, 0, 0, numpixels, numpixels)

    // draw 60 times / second
    requestAnimationFrame(drawWebcam)
}



// ******************************************************************************************
//
// startapp is called when the webcam successfully runs
//
// ******************************************************************************************
function getPixelColors() {

    let dataArray = []

    // read the tiny canvas colors using context.getImageData()
    /*
    for (let y = ...) {
        for (let x = ...) {
            // read canvas pixel color
            let pixelcolor = context.getImageData(...,..., 1, 1).data

            // use helper function to convert r,g,b to one value
            // let decimalColor = 

            // now add the number to the data array
        }
    }
    */
    
    return dataArray
}

// ******************************************************************************************
//
// train the neural network when a button has been pressed
//
// ******************************************************************************************
function trainNetwork(e) {
    console.log("train the network for button " + e.target.id)
}

// ******************************************************************************************
//
// start continuously classifying the webcam
//
// ******************************************************************************************
function startClassification(){
    console.log("start checking the webcam")
    // example to start an interval that classfies the webcam every 2 seconds
    // intervalid = setInterval(() => classifyWebcam(), 2000)
}

function classifyWebcam() {
    let dataArray = analysePixelData()
    // to do: test data against the neural network
    // let result = ...
    // if (result == ...) {
    //   console.log("Wave back!")
    // }
}