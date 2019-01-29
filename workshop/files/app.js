'use strict';

const video = document.querySelector('video')
const large_canvas = document.querySelector('#mosaic')
const large_context = large_canvas.getContext('2d')
const tiny_canvas = document.querySelector('#tiny_canvas')
const tiny_context = tiny_canvas.getContext('2d')
const display = document.querySelector('#display')

let width
let height
let intervalid

// how many pixels do we want to sample in each row and column
let numpixels = 10

// create the neural network
const net = new brain.NeuralNetwork()
let trainingData = []

// ******************************************************************************************
//
// start the webcam stream. function is in helper.js
//
// ******************************************************************************************
startWebcam()


// ******************************************************************************************
//
// startapp is called by the startWebcam function
//
// ******************************************************************************************
function startApp() {
    let wavebutton = document.getElementById("wave")
    let emptybutton = document.getElementById("empty")
    let personbutton = document.getElementById("person")

    wavebutton.addEventListener("click", (e) => addWaveData(e))
    emptybutton.addEventListener("click", (e) => addEmptyData(e))
    personbutton.addEventListener("click", (e) => addPersonData(e))

    let startbutton = document.getElementById("start")
    startbutton.addEventListener("click", (e) => startTraining(e))

    width = video.offsetWidth
    height = video.offsetHeight

    large_canvas.width = width
    large_canvas.height = height

    tiny_canvas.width = numpixels
    tiny_canvas.height = numpixels

    large_context.mozImageSmoothingEnabled = tiny_context.mozImageSmoothingEnabled = false
    large_context.webkitImageSmoothingEnabled = tiny_context.webkitImageSmoothingEnabled = false
    large_context.imageSmoothingEnabled = tiny_context.imageSmoothingEnabled = false

    // start drawing the webcam stream into the canvas so we can sample the colors
    drawWebcam()
}

// ******************************************************************************************
//
// convert the webcam stream to a canvas image
//
// ******************************************************************************************
function drawWebcam() {
    // Drawing the video twice
    large_context.drawImage(video, 0, 0, numpixels, numpixels)
    tiny_context.drawImage(video, 0, 0, numpixels, numpixels)

    // blow up one canvas just to see what our enlarged pixels look like
    large_context.drawImage(large_canvas, 0, 0, numpixels, numpixels, 0, 0, width, height)

    // draw 60 times / second
    requestAnimationFrame(drawWebcam)
}

// ******************************************************************************************
//
// convert pixels to array. use the tiny canvas and a x,y loop
//
// ******************************************************************************************
function analysePixelColors() {
    // TO DO 
    // loop through the rows and columns of the tiny canvas image to find the colors

    // example to find color at position 4,11
    let pixelcolor = tiny_context.getImageData(4, 11, 1, 1).data

    // TO DO 
    // return an array with pixel data - example code
    return [0,0.34,0.4,0.21,0.1,0,0.2,0,87]
}

// ******************************************************************************************
//
// add data for training
//
// ******************************************************************************************
function addWaveData(e) {
    // TO DO: create a training data object with the current image, labelled "wave"
    // TO DO: add this object to the trainingData array
    // example:
    // { input: [0, 0, 1], output: { wave: 1 } }
}

function addPersonData(e) {
}

function addEmptyData(e) {
}

// ******************************************************************************************
//
// start continuously classifying the webcam
//
// ******************************************************************************************
function startTraining() {
    // disable the buttons
    document.getElementById("wave").classList.add("disabled")
    document.getElementById("person").classList.add("disabled")
    document.getElementById("empty").classList.add("disabled")
    document.getElementById("start").classList.add("disabled")

    console.log(trainingData)

    // now train the neural network
    // net.train(trainingData)

    // start analysing the webcam feed live
    // intervalid = setInterval...
}

function classifyWebcam() {
    // TODO: get the current colors from the webcam
    // TODO: use net.run() to analyse the colors
    // TODO: respond to the result!

    // example
    // display.innerHTML = `CHANCE THAT SOMEONE IS WAVING: ${result.wave}`
}

