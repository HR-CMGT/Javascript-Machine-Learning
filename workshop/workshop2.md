# Machine Learning Workshop

- [Introduction and reading list](../README.md)
- [Workshop part 1 - Using a pre-trained Model](./workshop1.md)
- [Workshop part 2 - Training a model](./workshop2.md)
- [Workshop part 3 - Preparing data](./workshop3.md)

# Workshop Part 2 - Training a model

## Working with BrainJS

We are going to work with BrainJS. This javascript library creates a Neural Network that can analyse data.

For this part of the workshop, you can either work in a HTML document with an app script, or you can work in [Node](https://nodejs.org/en/), and run the app script from there.

### Working in the browser

Create a folder and make a new HTML and JS document.

**HTML**
```
<script src="https://cdn.rawgit.com/BrainJS/brain.js/master/browser.js"></script>
<script src="./app.js"></script>
```
**Javascript**
```
window.addEventListener("load", () => startApp())

function startApp(){
    const net = new brain.NeuralNetwork()
    console.log("created a neural net")
}
```

### Working in Node

Create a folder and download the BrainJS modules. 

```
npm install brain.js
```
Then create an app.js file
```
const brain = require('brain.js')
const net = new brain.NeuralNetwork(config)
console.log("created a neural net")
```
Run the app by typing `node app.js` in the terminal.

# BrainJS basics

You can train the neural network with the `train` command:
```
network.train(input:data, output:label)
```
The more variations of training data you supply, the more accurate the output will be. In this example we have four cat drawings that we want to classify as cats.
```
network.train(input:catdata1, output:"cat")
network.train(input:catdata2, output:"cat")
network.train(input:catdata3, output:"cat")
network.train(input:catdata4, output:"cat")
```
Now that our network knows what a cat is, we can supply unknown data and ask what this is:

```
let result = network.run(data)
```

## Predicting a soccer match

To use a neural network you have to learn to supply data as one-dimensional arrays.

In this example we will train a neural network with the results of previous football matches. Based on these results the network will predict the result of a future match. Let's say the result of the previous football matches has been:

- Team 1 versus team 2: Team 2 won
- Team 1 versus team 3: Team 3 won
- Team 2 versus team 3: Team 2 won
- Team 2 versus team 4: Team 4 won

By providing our network with these matches as arrays, we can train it. As output, we will use the position of the team that won. Output 0 means the first team won, output 1 means the other team won.

```
const network = new brain.NeuralNetwork()
network.train([
    { input: [1,2], output: [1] },  // team 2 wins
    { input: [1,3], output: [1] },  // team 3 wins
    { input: [2,3], output: [0] },  // team 2 wins
    { input: [2,4], output: [1] }   // team 4 wins
])
```
Finally, we can run a new match and get the expected result!
```
const prediction = network.run([1,1])
console.log(`probability is: ${prediction}`)    
```

# LSTM Neural Network

The *Long Short Term* neural net works well for interpreting *sequential* data, such as sentences, drawings, or musical melodies. 

In this example we will use an LSTM to interpret commands for a smart home. We will start by supplying as many variations for our home automation as possible. This is just an example, to make it work well you need lots more training data!

```
const network = new brain.recurrent.LSTM()
let trainingdata = [
    { input: 'Switch on the lights please', output: 'light' },
    { input: 'Turn the lights on', output: 'light' },
    { input: 'Can someone switch the lights on?', output: 'light' },
    { input: 'I'd like some music', output: 'music' },
    { input: 'Let's hear some music', output: 'music' }
]
```
When training a neural network, you can supply a number of iterations to make the predictions more accurate:

```
network.train(trainingdata, {
    iterations:2000
})
```
Now, when we have new user input, we can interpret the meaning:

```
const meaning = network.run('I'd like a little more light')
console.log(`Home system command: ${meaning}`)
```

## Saving the model

You don't want to train a model every time a user starts an app. In BrainJS, you can view the trained model data as JSON, so you can save it to a file.

```
var trainingData = network.toJSON()
console.log(trainingData)
```

## Loading the model

If you have loaded data from a JSON file, you can train the network with this data:
```
network.fromJSON(trainingData);
```

## Continue

[Continue with part 3 of the workshop](./workshop3.md)

## Documentation

- [BrainJS documentation and configuration](https://github.com/BrainJS/brain.js)
- [Example to recognise letters](https://github.com/BrainJS/brain.js/blob/master/examples/which-letter-simple.js)