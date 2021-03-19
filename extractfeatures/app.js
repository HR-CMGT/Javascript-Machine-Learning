const video = document.getElementById('webcam')
const featureExtractor = ml5.featureExtractor('MobileNet', modelLoaded)
const label = document.getElementById("label")
let classifier

const nomaskbtn = document.querySelector("#nomask")
const maskbtn = document.querySelector("#mask")
const trainbtn = document.querySelector("#train")

nomaskbtn.addEventListener("click", () => addNoMask())
maskbtn.addEventListener("click", () => addMask())
trainbtn.addEventListener("click", () => train())

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
    classifier = featureExtractor.classification(video, videoReady)
}

function videoReady(){
    console.log(classifier)
}

function addNoMask(){
    classifier.addImage(video, "draagt geen masker", addedImage)
}

function addMask() {
    classifier.addImage(video, "draagt masker", addedImage)
}

function train(){
    console.log("start training...")
    classifier.train((lossValue) => {
        console.log(lossValue)
        if(lossValue == null){
            startClassifying()
        }
    })
}

function startClassifying(){
    setInterval(()=>{
        classifier.classify(video, (err, result)=>{
            if(err) console.log(err)
            console.log(result)
            label.innerHTML = result[0].label
        })
    }, 1000)
}

function addedImage(){
    console.log("added image to network")
}