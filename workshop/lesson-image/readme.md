# Loading a Model for classifying

## Assignment

First, load TensorFlowJS, and the `mobileNet` model that has been trained on images

```html
<script src="https://cdnjs.cloudflare.com/ajax/libs/tensorflow/0.12.7/tf.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@0.1.1"></script>
```

Open the html page in a `localhost` server. If you don't have a development environment you can recreate this project on `https://codesandbox.io`

## Image

Include one or more images on your page

```html
<img id="image" src="./retriever.jpg">
```

## Load the model

```javascript
let model

async function loadModel() {
    model = await mobilenet.load()
    console.log("finished loading...")
    console.log("you can now use the model for classifying")
}

loadModel()
```

## Classify your image

After the model has loaded, you can call this function to start classifying images. Note that you only need to load the model once!

```javascript
async function classifyImage() {
    let img = document.getElementById("image")
    let predictions = await model.classify(img)
    console.log(predictions)
}
```

## Speak the result out loud!

Use this code to speak the first result of the classification out loud! Check the console to see what part of the result should be spoken.

```javascript
function speak() {
    let msg = new SpeechSynthesisUtterance()

    msg.text = "I think this photo shows a capybara"

    let selectedVoice = ""
    if (selectedVoice != "") {
        msg.voice = speechSynthesis.getVoices().filter(function (voice) { return voice.name == selectedVoice; })[0];
    }

    window.speechSynthesis.speak(msg)
}
```

## Improving the code

To test different images, we are constantly reloading the whole model. Add a `<button>` that can update the image with a new image and then classify it again.