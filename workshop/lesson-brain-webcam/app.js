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
let numpixels = 10
let trainingData = []

// ******************************************************************************************
//
// start the webcam stream. function is in helper.js
//
// ******************************************************************************************

// TODO CREATE A NEURAL NETWORK FIRST :)
// const net = ...

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
    let imageData = []

    for (let y = 0; y < numpixels; y++) {
        for (let x = 0; x < numpixels; x++) {
            // read pixel color r,g,b from the small canvas
            let pixelcolor = tiny_context.getImageData(x, y, 1, 1).data

            // convert r,g,b to a single digit with range 0 - 1
            let grey = rgbToGrayscale(pixelcolor[0], pixelcolor[1], pixelcolor[2])
            imageData.push(grey)
        }
    }

    return imageData
}

// ******************************************************************************************
//
// add data for training
//
// ******************************************************************************************
function addWaveData(e) {
    console.log("add example data for wave") 
}
function addPersonData(e) {
    console.log("add example data for person")
}
function addEmptyData(e) {
    console.log("add example data for empty")
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

    console.log("train the network and then start classifying")
}

function classifyWebcam() {
    console.log("classify the webcam colors after training is complete")
}

