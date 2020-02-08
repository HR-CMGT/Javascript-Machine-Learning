# Prediction and Smart Home with Brain JS

Let's look at Brain JS basics first. Create a neural network:

```javascript
const net = new brain.NeuralNetwork()
```

You can train the neural network with the `train` command. In this case we want the brain to understand that either one of the numbers has to be 1, but not both. 

```javascript
net.train([
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
])
```
Now, we can ask the machine to give a result for `1,0`
```javascript
const output = net.run([1, 0]) // [0.987]
```

## Predicting a soccer match

In the next example we will train a neural network with the results of previous football matches. Based on these results the network will predict the result of a future match. Let's train the network with previous match results:

- Team 1 versus team 2: Team 2 won
- Team 1 versus team 3: Team 3 won
- Team 2 versus team 3: Team 2 won
- Team 2 versus team 4: Team 4 won

We will use two team numbers as input. Output 0 means the first team won, output 1 means the other team won.

```javascript
const network = new brain.NeuralNetwork()
network.train([
    { input: [1,2], output: [1] },  // team 2 wins
    { input: [1,3], output: [1] },  // team 3 wins
    { input: [2,3], output: [0] },  // team 2 wins
    { input: [2,4], output: [1] }   // team 4 wins
])
```
Now, we can predict who will win if team 1 plays against team 4

```javascript
const prediction = network.run([1,4])
console.log(prediction)   // 0 means team 1 might win 
                          // 1 means team 4 might win! 
```
---
<br>
<br>

# Smart Home

The *Long Short Term* Neural Network works well for interpreting *sequential* data, such as sentences, drawings, or melodies. 

```javascript
const network = new brain.recurrent.LSTM()
```

In this example we will teach the machine to understand commands for a smart home. Create as many variations for turning the lights on and off as you can think of! Can you add commands for music and other smart home applications?

```javascript
let trainingdata = [
    { input: 'Switch on the lights please', output: 'light' },
    { input: 'Turn the lights on', output: 'light' },
    { input: 'Can someone switch the lights on?', output: 'light' }
]
```
When training the network, you can supply a number of iterations to make the predictions more accurate:

```javascript
network.train(trainingdata, { iterations:2000 })
```
Now, when we have new user input, we can interpret the meaning:

```javascript
const meaning = network.run('I would like some light')
console.log(meaning)   // LIGHT!
```

## Saving and loading the model

You don't want to train a model every time a user starts an app. In BrainJS, you can view the trained model data as JSON, so you can save it to a file.

```javascript
var trainingData = network.toJSON()
console.log(trainingData)
```
Once you loaded data from a JSON file, you can recreate the network:
```javascript
network.fromJSON(trainingData)
```

## Speak a reply

You can use the speak function to speak back to the user.

```javascript
function speak() {
    let msg = new SpeechSynthesisUtterance()

    msg.text = "Turning on the lights!"

    let selectedVoice = ""
    if (selectedVoice != "") {
        msg.voice = speechSynthesis.getVoices().filter(function (voice) { return voice.name == selectedVoice; })[0];
    }

    window.speechSynthesis.speak(msg)
}
```

## Listen to the microphone

Can you use the microphone to listen to spoken commands, instead of typing commands? Follow this tutorial to listen to the microphone!

- [Web Speech API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API)



# Links

- [BrainJS documentation](https://github.com/BrainJS/brain.js#brainjs)
- [Web Speech](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API)


