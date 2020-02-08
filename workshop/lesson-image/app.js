let model

speak()

async function loadModel() {

}

async function classifyImage() {

}

function speak() {
    let msg = new SpeechSynthesisUtterance()
    msg.text = "This photo shows a capybara"
    
    let selectedVoice = ""
    if (selectedVoice != "") {
        msg.voice = speechSynthesis.getVoices().filter(function (voice) { return voice.name == selectedVoice; })[0];
    }
    
    window.speechSynthesis.speak(msg)
}