# Introduction

- [Training, algorithms, and models](#basics)
- [Machine Learning problems](#disciplines)
- [Coding with Javascript](#javascript)
  - Brain.js
  - ml5.js
  - Tensorflow.js

<br>
<br>
<br>

# <a name="basics"></a>Training, algorithms, and models

In traditional programming, a programmer writes a *set of instructions*, that is executed by the computer. 

In Machine Learning, we let the computer find the optimal decision by itself. We provide **training data** to an **algorithm** and we receive a **trained model**. The model can make decisions for us!

In this example, we train a model to recognise cats based on an existing set of cat drawings:

![model1](./images/model1.png)

When we have a model, we can check if any new drawing is a cat or not! We can even ask it to generate a new cat drawing for us!

![model2](./images/model2.png)



<br>
<br>
<br>
<br>

# <a name="disciplines"></a>Machine Learning problems

![pose](./images/pose.png)

Depending on your data and your goal, you can use different approaches for your Javascript ML project.

| Problem                                | Approach                                                                                              | Javascript example                               |
|-------------------------------------|-------------------------------------------------------------------------------------------------------|--------------------------------------------------|
| Find patterns in a simple excel sheet   | Use K-Nearest-Neighbour to find patterns in arrays of numbers                                         | [kNear](https://github.com/NathanEpstein/KNear), [KNNClassifier](https://ml5js.org/reference/api-KNNClassifier/)                             |
| Find patterns in a complex excel sheet  | Use a basic Neural Network to find patterns in arrays of numbers                                      | [brainJS](https://github.com/BrainJS/brain.js#for-training-with-neuralnetwork), [ML5 Neural Network](https://learn.ml5js.org/docs/#/reference/neural-network), [Tensorflow Basics](https://www.tensorflow.org/js/tutorials/training/linear_regression)             |
| Understand meaning of text | Use LSTM Neural Network or Word2Vec to find meaning in sentences                                      | [BrainJS LSTM](https://github.com/BrainJS/brain.js#for-training-with-rnn-lstm-and-gru), [Word2Vec](https://learn.ml5js.org/docs/#/reference/word2vec)                                |
| Understand sentiment in text | Use existing sentiment model                                      | [ML5 sentiment](https://ml5js.org/reference/api-Sentiment/), [TensorFlow sentiment](https://github.com/tensorflow/tfjs-examples/tree/master/sentiment), [Detect Comment Toxicity](https://storage.googleapis.com/tfjs-models/demos/toxicity/index.html)                                |
| Recognise body poses                | Use an existing pre-trained body pose model                                                           | [Train a Pose Model with Teachable Machine](https://teachablemachine.withgoogle.com) [ML5 PoseNet](https://learn.ml5js.org/docs/#/reference/posenet)                                    |
| Recognise objects in images         | Use an existing pre-trained image model, or train your own model using a Convolutional Neural Network | [Train a model with Teachable Machine](https://teachablemachine.withgoogle.com), [ML5 YOLO](https://learn.ml5js.org/docs/#/reference/yolo), [Tensorflow Object Detection](https://github.com/tensorflow/tfjs-examples/tree/master/simple-object-detection)            |
| Recognise hand written text         | Use the MNIST model | [Tensorflow MNIST](https://github.com/tensorflow/tfjs-examples/tree/master/mnist)            |
| Recognise facial expressions        | Use an existing facial expression model                                                   | [Face-API](https://github.com/justadudewhohacks/face-api.js) |
| Generate text or images             | Use a Recurrent Neural Network                                                                        | [ML5 Sketch RNN](https://learn.ml5js.org/docs/#/reference/sketchrnn), [BrainJS RNN](https://github.com/BrainJS/brain.js#for-training-with-rnn-lstm-and-gru)                              |


---
<br>
<br>

![brain](./images/brain.png)

# <a name="javascript"></a>Coding with Javascript

Javascript allows us to publish our projects online, and provides easy ways to visualise our results using html and css.

## Brain.JS

[BrainJS](https://github.com/BrainJS/brain.js) is a library that allows you to instantiate a Neural Network, train it and run a classification in just a few lines of code. This example learns if text on a RGB background should be white or black:

```javascript
const net = new brain.NeuralNetwork()

net.train([
  { input: { r: 0.03, g: 0.7, b: 0.5 }, output: { black: 1 } },
  { input: { r: 0.16, g: 0.09, b: 0.2 }, output: { white: 1 } },
  { input: { r: 0.5, g: 0.5, b: 1.0 }, output: { white: 1 } },
])

const output = net.run({ r: 1, g: 0.4, b: 0 }) // { white: 0.99, black: 0.002 }
```

- [Source code and examples for BrainJS](https://github.com/BrainJS/brain.js)
- [Youtube BrainJS introduction](https://www.youtube.com/watch?v=RVMHhtTqUxc)
- [Recognise letters](https://github.com/BrainJS/brain.js/blob/master/examples/which-letter-simple.js)
- [Recognise a drawing](https://output.jsbin.com/mofaduk) and [code](https://gist.github.com/mac2000/fc54e6d6bdcbfde28b03dc2a43611270)
- [Live code example using table data](https://scrimba.com/c/c36zkcb)

## ML5.JS

ML5 supplies a simplified wrapper with clear documentation and examples for many existing Machine Learning libraries, such as TensorFlow and YOLO. In this example, we teach the machine what is left and what is right:

```javascript
let nn = ml5.neuralNetwork({
  inputs: 1,
  outputs: 2,
  task: 'classification',
  debug: true
})

nn.addData( 100,  ['left'])
nn.addData( 600,  ['right'])
nn.addData( 150,  ['left'])
nn.addData( 800,  ['right'])

nn.normalizeData()
nn.train(finishedTraining)

function finishedTraining(){
  nn.classify([160], (err, result) => console.log(result)) // LEFT
}
```

- [Introduction to the ML5 library](https://ml5js.org)
- [ML5 Neural Network](https://learn.ml5js.org/docs/#/reference/neural-network)
- [ML5 Image Recogition](https://learn.ml5js.org/docs/#/reference/yolo)
- [ML5 Pose Recogition](https://learn.ml5js.org/docs/#/reference/posenet)



## Tensorflow JS

TensorFlow is Google's Neural Network library. TensorFlow is available for Javascript, Python and IoT devices. In TensorFlow you can build your own custom Neural Network to serve many different purposes.

- [Tensorflow for Javascript](https://www.tensorflow.org/js/)
- [Load an existing model and classify an image in 3 lines of code](./workshop/workshop3.md)
- [Tensorflow JS Tutorials](https://www.tensorflow.org/js/tutorials)
- [Tensorflow Neural Network](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#0)
- [Hello World in Tensorflow.JS](https://meowni.ca/posts/hello-tensorflow/)
- [Audio example](https://codelabs.developers.google.com/codelabs/tensorflowjs-audio-codelab/), [Webcam example](https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html)
- [Apply Anime Drawing Style on your images](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html)
- [Detect toxicity in online comments](https://storage.googleapis.com/tfjs-models/demos/toxicity/index.html)

<br>
<br>

