'use strict'

const data = document.querySelector('data')
const label = document.querySelector('#label')
const video = document.querySelector('video')
const canvas = document.querySelector('#mosaic')
const context = canvas.getContext('2d')
const trainBtn = document.querySelector("#train")
const classifyBtn = document.querySelector("#classify")
const result = document.querySelector("#result")

const numpixels = 10
let width// 340 
let height// 255
let intervalid
let dataArray

function initSettings() {
    width = video.offsetWidth
    height = video.offsetHeight

    canvas.width = width
    canvas.height = height

    context.mozImageSmoothingEnabled = false
    context.webkitImageSmoothingEnabled = false
    context.imageSmoothingEnabled = false

    // start taking webcam snapshots
    webcamSnapshot()

    // create array of pixel values every second
    intervalid = setInterval(() => generatePixelValues(), 1500)

    // button functions
    trainBtn.addEventListener("click", () => trainMachine())
    classifyBtn.addEventListener("click", () => classifyData())
}



function webcamSnapshot() {
    // drawing the video very small causes pixelation. then blow up the canvas image itself
    context.drawImage(video, 0, 0, numpixels, numpixels)
    context.drawImage(canvas, 0, 0, numpixels, numpixels, 0, 0, width, height)

    // draw 60 times / second
    requestAnimationFrame(webcamSnapshot)
}

function generatePixelValues() {

    data.innerText = ""
    dataArray = []

    for (let pos = 0; pos < numpixels * numpixels; pos++) {
        let col = pos % numpixels
        let row = Math.floor(pos / numpixels)

        let x = col * (width / numpixels)
        let y = row * (height / numpixels)

        // get pixel colors from the grid
        let p = context.getImageData(x + width / 20, y + height / 20, 1, 1).data

        // convert three r,g,b values into one value
        let decimalColor = rgbToDecimal(p[0], p[1], p[2])
        dataArray.push(decimalColor)

        // show values
        data.innerText += decimalColor + ", "
    }

}

function trainMachine() {
    console.log("use dataArray values and label to train the machine")
    console.log(dataArray)
    console.log(label.value)
}

function classifyData() {
    console.log("use dataArray values to classify the image")
    result.innerHTML = "This is a capybara!"
}


//
// INITIALIZE THE APP BY STARTING THE WEBCAM
//
function initializeWebcam() {
    // docs: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia
    if (navigator.mediaDevices) {
        navigator.mediaDevices.getUserMedia({ video: true })
            // permission granted:
            .then(function (stream) {
                video.srcObject = stream
                video.addEventListener("playing", initSettings)
                //video.addEventListener('click', takeSnapshot)
            })
            // permission denied:
            .catch(function (error) {
                document.body.textContent = 'Could not access the camera. Error: ' + error.name
            })
    }
}

initializeWebcam()