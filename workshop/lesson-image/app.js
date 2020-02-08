let model
let img = document.getElementById("image")

async function loadModel() {
    model = await mobilenet.load()
    console.log("finished loading...")
    classifyImage()
}

async function classifyImage() {
    let predictions = await model.classify(img)
    console.log(predictions)
    speak(predictions[0].className)
}

function speak(prediction) {
    let msg = new SpeechSynthesisUtterance()

    msg.text = "I think this photo shows a " + prediction

    let selectedVoice = ""
    if (selectedVoice != "") {
        msg.voice = speechSynthesis.getVoices().filter(function (voice) { return voice.name == selectedVoice; })[0];
    }

    window.speechSynthesis.speak(msg)
}


console.log("start loading")
loadModel()