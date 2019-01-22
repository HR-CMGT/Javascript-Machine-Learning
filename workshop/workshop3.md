# Machine Learning Workshop

- [Introduction and reading list](../README.md)
- [Workshop part 1 - Using a pre-trained Model](./workshop1.md)
- [Workshop part 2 - Training a model](./workshop2.md)
- [Workshop part 3 - Preparing data](./workshop3.md)

# Workshop Part 3 - Preparing data

So far, we've used simple data to learn how to train a model. In reality, data is often complex. You will spend a lot of time preparing data before you can even start working with your neural net.

For the last part of the workshop, you will work by yourself or in a team, to prepare data and then train a model. Two starter projects are provided:

- **Webcam recognition** You will download a project that shows webcam output in the browser. There is a function to read pixeldata from the webcam. It's up to you to come up with a useful application and then train a model to recognise webcam output. 
- **Spoken command recognition** You will download a project that allows you to talk to the browser. Your words will appear as text. It's up to you to come up with a useful application and then train the model to recognise spoken commands. 

# Webcam recognition

You can download the HTML and JS files to get started. This will show you a webcam feed and an example of how to read canvas data. 

Read the explanations below and see if you can build an image recognition app!

## Preparing image data

You will need to convert the canvas image to a one-dimensional array of numbers, for example: `image = [3,5,2,6,3,6]`

Start by reading the colors of each row and column, using two `for` loops. 

A canvas color will consist of three values (red, green, blue). You could push these three values straight into the array, but we can also use a formula to reduce R,G,B to one single value. That way the array can stay just a bit smaller, making training faster.

Another way to prevent our array becoming huge, is to not read EVERY pixel of the canvas. An image of 100x100 pixels would  result in 10.000 values. That will make training slow.

The best way to get pixel data out of a large image is to resize the image down to 10x10 pixels. Resizing will make sure that the colors are a perfect average, and we'll only have 100 values in our array for each image.

## Starter files

The starter files contain the webcam stream and a resized smaller canvas. It also contains example code for creating the neural net, reading the canvas color and using the buttons.

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

Image recognition is a complicated subject. The above example won't really recognise specific features such as eyes, faces or other shapes. It just learns pixel colors. For more advanced image recognition you have to look at *feature detection*.

# Spoken command recognition

The idea is that you let your browser take action upon hearing spoken commands, just like a home automation system. You will first come up with a creative concept, and then create data and train a model. 

## Starter files

The starter project contains examples for browser-native text and speech manipulation, and initialising the Neural Network.

## TO DO

- Read the [MDN browser speech documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- Create text from speech when the user presses a *record* button, or by using native audio input for a form element. 
- Train the model with that text to be a certain label, for example `["computer, turn on the lights", "light"]`.
- Repeat this a number of times, until the model has been trained with enough examples
- When you press the *run* button, the same thing happens as above, but instead of training, you test the network for a result.
- You can show an image, play a sound, or use web speech to respond to the microphone input.

## Documentation

- [MDN browser speech documentation](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API)
- [LSTM neural net example]()

# Next steps

A great next step would be to look at the [ML5 library](https://ml5js.org), which contains many examples for different types of machine learning. ML5 uses TensorFlowJS.

The [reading list](../README.md) contains tons of links to documentation, libraries and tutorials.