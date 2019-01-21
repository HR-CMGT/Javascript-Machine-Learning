# Machine Learning Workshop

- [Introduction and reading list](../README.md)
- [Workshop part 1 - Using a pre-trained Model](./workshop1.md)
- [Workshop part 2 - Training a model](./workshop2.md)
- [Workshop part 3 - Preparing data](./workshop3.md)

# Workshop Part 2 - Training a model

#### Predicting a soccer match

In this example we will train a neural network with the results of previous football matches. Based on these results the network will predict the result of a future match. Let's say the result of the previous football matches has been:

- Team 1 versus team 2: Team 2 won
- Team 1 versus team 3: Team 3 won
- Team 2 versus team 3: Team 2 won
- Team 2 versus team 4: Team 4 won

We will first load the Neural Network library BrainJS:

`<script src="brain.js"></script>`

Then, we can instantiate the type of Neural Network that fits the problem we are trying to solve:

`const network = new brain.NeuralNetwork()`

Now, we can train our model with the results of previous matches:
```
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

#### Understanding spoken commands

The **LSTM** neural net works well for interpreting sequential data, such as sentences, drawings, or musical melodies. 

`const network = new brain.recurrent.LSTM()`

In this example we will use an LSTM to interpret commands for a smart home. We will start by supplying as many variations for our home automation as possible. This is just an example, to make it work well you need lots more training data!

```
var trainingdata = [
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
console.log(`Home system command: ${output}`)
```

#### Saving the model

You don't want to train a model every time a user starts an app. In BrainJS, you can view the model data as JSON, so you can save it to a file:

```
var trainingData = network.toJSON()
console.log(trainingData)
```
If you have data from a JSON file, you can train the network with this data:
```
network.fromJSON(trainingData);
```

## Workshop Part 3 - Understanding an algorithm

This algorithm uses the distance between points to classify a new point. The KNN algorithm is useful to understand the basics of training a model and classifying data, without a complex algorithm:

- [Using the 'k-nearest neighbour' algorithm to train a model](https://github.com/NathanEpstein/KNear)
- [Tutorial for 'K-Nearest-Neighbour'](https://github.com/KokoDoko/webcam-detectotron)