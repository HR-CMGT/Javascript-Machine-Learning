# ML5 Neural Network

The ML5 Library makes it easy to create a Neural Network for classification or regression. After training, you can save the model and load it in for classifying new data.

```html
<script src="https://unpkg.com/ml5@0.4.3/dist/ml5.min.js"></script>
```

## Basic classification Neural Network

Here we train a network to learn that small numbers mean `left` and big numbers mean `right`. 

Set `debug` to true to get a cool training visualisation!

```javascript
let nn = ml5.neuralNetwork({
    inputs: 1,
    outputs: 2,
    task: 'classification',
    debug:true
})

nn.addData([20], ['left'])
nn.addData([50], ['left'])
nn.addData([250], ['right'])
nn.addData([320], ['right'])

nn.normalizeData()
nn.train(finishedTraining)

function finishedTraining() {
    nn.classify([300], (err, result) => console.log(result))
}
```
### More inputs

Note that there is only one input. If our data has more inputs (for example, `x, y`), we can set `inputs` to `2` and then use `nn.addData([12,40], 'cat')` to train, and `nn.classify([30,50])` to classify.

## Saving the model

```javascript
nn.save("mymodel")
```
https://learn.ml5js.org/docs/#/reference/neural-network?id=save

## Loading a model

When loading a model, you can skip the `addData()`, `normalizeData()` and `train()` steps of creating a neural network!

```javascript
nn.load('path/to/model.json')
```
https://learn.ml5js.org/docs/#/reference/neural-network?id=load

## Codesandbox

To run this example on codesandbox, start a new Vanilla JS sandbox, and then add ML5 as a dependency, or you can just fork this example project: https://codesandbox.io/s/ml5-starter-vtb7q

## Documentation and examples

https://ml5js.org
https://learn.ml5js.org/docs/#/reference/neural-network
https://editor.p5js.org/ml5/sketches/NeuralNetwork_Simple_Classification

