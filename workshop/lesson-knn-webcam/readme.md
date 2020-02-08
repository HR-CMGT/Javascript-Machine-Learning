# K-Nearest-Neighbour Webcam Detection

We will classify webcam images using just K-Nearest-Neighbour

## Running the project

Open `index.html` in a `localhost` server, for example with the `live server` plugin for VS Code. 

### No editor?

You can create this project on `https://codesandbox.io`. Copy the code, or fork the example from https://codesandbox.io/s/workshop-knn-webcam-start-zsiwb

## Starter code

![screenshot](screenshot.png)

We need to provide the algorithm with webcam data. To do this. we draw the webcam in a canvas, so we can sample the color values of the pixels. Because even a small image of 400x300 pixels would have 120000 values, we first reduce this by pixelating the image.

Every second, the webcam image is pixelated and the color values are shown in the html page. 

## Assignment

Look at these two functions:

```javascript
function trainMachine() {
    console.log("use dataArray values and label to train the machine")
    console.log(dataArray)
    console.log(label.value)
}

function classifyData() {
    console.log("use dataArray values to classify the image")
    result.innerHTML = "This is a capybara!"
}
```

Add the same machine learning code as you did in the [previous example](../lesson-knn/readme.md).

```javascript
// previous practice example
machine.learn([10, 10, 10], 'cat')
let catOrDog = machine.classify([7,7,7])
```

Replace the cat and dog data with the color values from the webcam, and the label value from the form field.

Then connect the UI to your training code:

- Connect the `train` button to the KNN `trainMachine()` function with `addEventListener`.
- For training, use the pixel data in `dataArray` and the label from the text input field.
- Connect the `classify` button to code that calls the KNN `classifyData()` function. Pass the current `dataArray` from the webcam to the classify function.
- Now you can start training. Type a label in the field and for each label, train ~10 images. Then check if your classify button works!

## Github documentation

https://github.com/NathanEpstein/KNear