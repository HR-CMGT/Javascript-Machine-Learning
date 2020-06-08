const video = document.getElementById('webcam')
const featureExtractor = ml5.featureExtractor('MobileNet', modelLoaded)
const label = document.getElementById("label")
let classifier

if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream
        })
        .catch((err) => {
            console.log("Something went wrong!");
        });
}

function modelLoaded(){
    console.log("The mobileNet model is loaded!")
}