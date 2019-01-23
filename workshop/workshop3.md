# Machine Learning Workshop

- [Introduction](./introduction.md)
- [Part 1 - Using a pre-trained Model](./workshop1.md)
- [Part 2 - Training a model](./workshop2.md)
- [Part 3 - Preparing data](./workshop3.md)
- [Reading list](../README.md)

# Workshop Part 3 - Preparing data

So far, we've used simple data to learn how to train a model. In reality, data is often complex. You will spend a lot of time preparing data before you can even start working with your neural net.

For the last part of the workshop, you will work by yourself or in a team, to prepare data and then train a model with webcam data. There is a starter project provided. 

You can also choose to work on one of the alternative projects, where you have to figure everything out by yourself!

# Webcam recognition

You can download the HTML and JS files to get started. This will show you a webcam feed and an example of how to read canvas data. Read the explanations below and see if you can build an image recognition app!

## Preparing image data

You will need to convert the canvas image to a one-dimensional array of numbers, for example: `image = [3,5,2,6,3,6]`

Start by reading the colors of each row and column, using two `for` loops. 

A canvas color will consist of three values (red, green, blue). You could push these three values straight into the array, but we can also use a formula to reduce R,G,B to one single value. That way the array can stay just a bit smaller, making training faster.

Another way to prevent our array becoming huge, is to not read EVERY pixel of the canvas. An image of 100x100 pixels would  result in 10.000 values. That will make training slow.

The best way to get pixel data out of a large image is to resize the image down to 10x10 pixels. Resizing will make sure that the colors are a perfect average, and we'll only have 100 values in our array for each image.

## Starter files

The starter files contain the webcam stream, code for the buttons, and starter code for reading the canvas pixel data.

<a href="https://github.com/HR-CMGT/TLE3-machine-learning/tree/master/workshop/files" target="_blank">Get the starter files</a>

## TO DO

- Sample the canvas image when you press the *train* button. 
- Train the model with the array.
- Repeat this a number of times, until the model has been trained with enough examples
- When you press the *run* button, start a function that samples the webcam every second, using `setInterval`.
- Test that data against the network, and take an action when the data matches the training data.
- You can show an image, play a sound, or use [web speech](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
 to respond to the webcam. For example, when the computer recognises you, it could speak a greeting.

#### Inspiration

- This [code](https://gist.github.com/mac2000/fc54e6d6bdcbfde28b03dc2a43611270) shows how to [recognise a hand-drawn sketch](https://output.jsbin.com/mofaduk) using BrainJS
- The [Teachable machine](https://teachablemachine.withgoogle.com) shows different images after being trained with the webcam.

#### Notes

Image recognition is a complicated subject. The above example won't really recognise specific features such as eyes, faces or other shapes. It just learns pixel colors. For more advanced image recognition you have to look at *feature detection*

- [Feature detection tutorial with BrainJS](https://scrimba.com/c/c36zkcb)
- [Feature detection with ML5](https://ml5js.org/docs/custom-classifier)

# Alternative projects

These projects do not have starter code. You'll have to figure everything out by yourself, using the knowledge you've gained, and the powers of the internet.

## Speech recognition

- Read the [MDN browser speech documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API) to learn how to input and output spoken text.
- Train a LSTM neural net to recognise spoken commands, for example `["computer, turn on the lights", "light"]`

## Arduino sensor data recognition

- Read the [Johnny Five documentation](http://johnny-five.io) to learn how to read Arduino sensor data into a Node application.
- Train a network with data from the sensors to respond to certain events!

### Documentation

- [MDN browser speech documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- [Example to train LSTM neural net](https://github.com/bradtraversy/brainjs_examples/blob/master/02_hardware-software.js)
- [Johnny Five](http://johnny-five.io)

# Next steps

A great next step would be to look at the [ML5 library](https://ml5js.org), which contains many examples for different types of machine learning. Perhaps you can start [teaching the computer how to draw](https://www.youtube.com/watch?v=pdaNttb7Mr8)!

The [reading list](../README.md) contains tons of links to documentation, libraries and tutorials.