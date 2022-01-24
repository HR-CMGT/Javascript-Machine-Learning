# Face mask detection with Transfer Learning

If you used the [ML5 ImageClassifier](https://learn.ml5js.org/#/reference/image-classifier) you may have noticed it doesn't always recognise what you want it to recognise. 

Using the [Feature Extractor](https://learn.ml5js.org/#/reference/feature-extractor) we can re-train the model to recognise your own images.

<br>
<br>
<br>

## Feature extraction

![features](./features.png)

This term means we use a model that has learned **HOW** to look at images. It finds the most important defining features. With this knowledge, it can find the distinguishing features in our own images as well.

[Watch this super cool visualisation of feature extraction](https://www.youtube.com/watch?v=f0t-OCG79-U)

<br>
<br>
<br>

## HTML

First include ML5, and include a video tag in your html

```html
<script src="https://unpkg.com/ml5@0.4.3/dist/ml5.min.js"></script>
<video autoplay playsinline muted id="webcam" width="533" height="300"></video>
```
In javascript, switch on the webcam with
```javascript
if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => {
            video.srcObject = stream
        })
        .catch((err) => {
            console.log("Something went wrong!");
        });
}
```
<br>
<br>
<br>

## Add training data

Create the featureExtractor, and in the callback you can create the classifier. 

```javascript
const featureExtractor = ml5.featureExtractor('MobileNet', modelLoaded)
const video = document.getElementById('webcam')
function modelLoaded() {
    console.log('Model Loaded!')
    classifier = featureExtractor.classification(video, videoReady)
}
function videoReady(){
    console.log("the webcam is ready")
}
```
Create buttons for all your labels in your HTML page.
```html
<button id="mask">Wearing a mask</button>
```
When you click a button, you can add the current webcam image with that label as a training image:
```javascript
const maskbtn = document.getElementById("mask")
maskbtn.addEventListener("click", () => addMaskImage())

function addMaskImage() {
    classifier.addImage(video, 'wearing a mask', ()=>{
        console.log("added image to model!")
    }))
}
```
The callback is just to check if the image was succesfully added to the model.

<br>
<br>
<br>

## Training

After adding about 10-20 images for each label, you can call the training function. The loss value should be getting smaller while your network is learning.
```javascript
classifier.train((lossValue) => {
    console.log('Loss is', lossValue)
    if(lossValue == null) console.log("Finished training")
})
```
<br>
<br>
<br>

## Classifying

When the lossvalue becomes `null`, you can start an interval that checks the webcam every second!
```javascript
label = document.getElementById("label")

setInterval(()=>{
    classifier.classify(video, (err, result) => {
        if (err) console.log(err)
        console.log(result)
        label.innerHTML = result[0].label
    })
}, 1000)
```

<br>
<br>
<br>

## Saving and loading the trained model

You don't want to `train()` a model every time a user starts an app. [Use the ML5 save and load options to load your own trained model](https://learn.ml5js.org/docs/#/reference/feature-extractor?id=save).


## Regression

Instead of classifying an image into class A,B or C, you might want to get a value from 0 to 100. For example, to assess how damaged a car door is.

[regression](https://learn.ml5js.org/docs/#/reference/feature-extractor?id=regression) and [prediction](https://learn.ml5js.org/docs/#/reference/feature-extractor?id=predict)

<br>
<br>
<br>

---

## Links

- [ML5.JS](https://ml5js.org/)
- [ML5 Feature Extractor documentation](https://learn.ml5js.org/docs/#/reference/feature-extractor)
- [ML5 Plain Javascript example code](https://github.com/ml5js/ml5-library/tree/main/examples/javascript/FeatureExtractor/FeatureExtractor_Image_Classification)
- [Coding Train Tutorial Feature Extractor](https://www.youtube.com/watch?v=eeO-rWYFuG0)

