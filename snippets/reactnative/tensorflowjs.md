# TensorflowJS in React Native

Volledige code voor TensorflowJS in React Native.

- Inlezen CSV
- Trainen van een model
- Doe een voorspelling

<br>

### COMPONENT

Het component laadt de CSV en roept de training functie aan. Je kan de predict functie aanroepen met een button.

```js
import * as React from 'react';
import { Text, View, Button, StyleSheet, TextInput } from 'react-native';
import { readRemoteFile, readString } from 'react-native-csv';
import * as tf from '@tensorflow/tfjs'
import { createNeuralNetwork } from "./createNeuralNetwork.js"
import carsData from '../assets/cars.csv';

export default function TFExample() {
  const [prediction, setPrediction] = React.useState(0);
  const [hp, onChangeHP] = React.useState(50);
  const [weight, onChangeWeight] = React.useState(2000);
  const machine = React.useRef();
  const normalizeValues = React.useRef()


  //
  // prediction
  //
  const makePrediction = () => {
      // let op dat hp en weight numbers zijn en geen strings
      const userInput = tf.tensor2d([[hp,weight]])
      // normalize the user input, un-normalise the prediction
      const nc = normalizeValues.current
      const normInput = userInput.sub(nc.inputMin).div(nc.inputMax.sub(nc.inputMin))
      const predictionTF = machine.current.predict(normInput)
      const unnormalisedPredictionTF = predictionTF.mul(nc.labelMax.sub(nc.labelMin)).add(nc.labelMin)
      const prediction = Math.round(unnormalisedPredictionTF.dataSync()[0])

      setPrediction(prediction)
  }

  //
  // load csv
  //
  React.useEffect(() => {
    const loadCSV = () => {
      readRemoteFile(carsData, {
        download: true,
        header: true,
        dynamicTyping: true,
        complete: (results) => {
          csvLoaded(results.data);
        },
      });
    };

    const csvLoaded = async(data) => {
        data.sort(() => (Math.random() - 0.5))

        const inputs = data.map(d => [d.horsepower, d.weight])
        const outputs = data.map(d => d.mpg)
        const [model, normValues] = await createNeuralNetwork(inputs, outputs)
        
        machine.current = model
        normalizeValues.current = normValues
    };

    loadCSV();
  }, []);
}
```



<Br><Br><Br>



## Het Neural Network bouwen

Voor het overzicht is het bouwen van het neural network en het normalizen van de data in een eigen JS bestand geplaatst. Deze moet je `importeren` in het component. 


```js
import * as tf from '@tensorflow/tfjs'

export const createNeuralNetwork = async (inputs, outputs) => {
    // tensors maken van de inputs en outputs, let op dat alle data NUMBERS zijn!
    const inputTensor = tf.tensor2d(inputs)
    const labelTensor = tf.tensor1d(outputs)

    const [inputMax, inputMin, labelMax, labelMin] = [inputTensor.max(0,false), inputTensor.min(0,false), labelTensor.max(), labelTensor.min()]
    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

    // bouw het model met 2 features: horsepower, weight
    const numFeatures = inputs[0].length
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 8, inputShape: [numFeatures] }))
    model.add(tf.layers.dense({ units: 1 }))
    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })
    // aantal epochs instellen
    await model.fit(normalizedInputs, normalizedLabels, { epochs: 5 })
    return [model, { inputMin, inputMax, labelMin, labelMax }]    
}
```


- [Expo Snack Voorbeeld](https://snack.expo.dev/@eerk/tensorflow-neural-network)

<br><br><br>

## Training visualisatie

Je kan de visualisatietool van tensorflowJS niet gebruiken in React Native omdat de tool verwacht dat er een browser venster is. Als het niet nodig is om live te trainen met gebruikersinput zou je het model al vantevoren kunnen maken in de browser.

<br><br><br>

## Model opslaan

Je kan het model opslaan in de `AsyncStorage` van React Native. De `tfjs-react-native` library heeft hier een helper functie voor:

- [Tensorflow React Native Storage](https://js.tensorflow.org/api_react_native/0.6.0/#asyncStorageIO)