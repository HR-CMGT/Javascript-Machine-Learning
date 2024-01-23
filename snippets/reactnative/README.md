# Machine Learning in React Native

In deze readme vind je de werkwijze voor het toepassen van een aantal algoritmes uit de PRG8 lessen in React Native:

- Workflow
- KNN
- CSV laden
- Decision Tree
- Neural Networks
- Image Prediction
- ChatGPT Assistent

<br><br><br>

## Workflow

- Het laden van data en trainen van een model wil je maar 1 keer doen. Dit doe je in de `useEffect()` functie. Deze functie wordt 1 keer uitgevoerd bij het laden van de app.
- Ditzelfde geldt voor het laden van een getrained model.
- Het model zelf moet bewaard blijven in het component, zodat je later de voorspellingen kan uitvoeren. 
- Dit kan je doen in een `state`, maar omdat state continu gemonitord wordt door React, onthouden we het model met `useRef`.
- Javascript code die gebruik maakt van `window`, HTML elementen, CSS, `<script>` tags, of andere browser eigenschappen kan je niet gebruiken in React Native. 
- Sommige libraries hebben een React Native variant.

## Trainen van een model

Je hoeft het trainen van een model niet persé in React Native te doen. Meestal zal je een getrained model inladen. De code blijft hetzelfde, je kan zelf bepalen op welk moment je de training wil doen. 

<br>

### Pseudocode

```js
import { useState, useEffect, useRef } from 'react';

export function App {
    const model = useRef(null)
    
    React.useEffect(() => {
        // Load data
        // Train model
        // Store model in useRef
    }, [])
    
    const predict = () => {
        // Predict
    }
    
    return (
        <View>
            <Button onPress={predict} />
        </View>
    );
}
```

<br><br><br>

# KNN

Maak een online React Native testomgeving aan met [Snack](https://snack.expo.dev/). Installeer KNN:

```bash
npm install knear
```
We passen bovenstaande pseudocode toe, met een kleine hoeveelheid testdata die in de code zelf staat. 

```js
import * as React from 'react';
import { Text, View, Button, StyleSheet, TextInput } from 'react-native';
import knn from 'knear';

export default function KNNExample() {

  const machine = React.useRef(new knn.kNear(2))

  const makePrediction = () => {
        const pred = machine.current.classify([5, 5, 3])
        console.log(pred)
  }

  React.useEffect(() => {
      machine.current.learn([1, 2, 3], 'cat')
      machine.current.learn([0, 0, 0], 'cat')
      machine.current.learn([3, 1, 3], 'cat')
      machine.current.learn([10, 10, 10], 'dog')
      machine.current.learn([14, 10, 9], 'dog')
      machine.current.learn([9, 12, 13], 'dog')
  }, [])

  return (
    <View>
      <Text>Predict cats or dogs based on body features</Text>
      <Button onPress={makePrediction} title="Predict!" />
    </View>
  );
}
```
<br>

## Invulveld en resultaat tonen

Je hebt de waarde uit een invulveld nodig om te weten wat de gebruiker wil voorspellen. Ook wil je het resultaat van de voorspelling tonen aan de gebruiker. Hiervoor kan je `state` gaan bijhouden.

```js
const [prediction, setPrediction] = React.useState('');
const [weight, onChangeWeight] = React.useState(0);

const makePrediction = () => {
    const pred = machine.current.classify([weight])
    setPrediction(pred)
}
return (
    <View>
        <TextInput
          keyboardType='numeric'
          onChangeText={text => onChangeWeight(parseInt(text))}
          value={weight}
        />
        <Button onPress={makePrediction} title="Predict!" />
        <Text>I think it's a {prediction} !</Text>
    </View>
)
```

Bekijk deze snack voor het complete voorbeeld.

- [Online Voorbeeld](https://snack.expo.dev/@eerk/knn-snack?)

<br><br><br>

# CSV laden

In dit voorbeeld gebruiken we de [React Native variant van Papa Parse](https://react-native-csv.js.org) om CSV data te laden. 

```
npm install react-native-csv

import { readRemoteFile, readString } from 'react-native-csv';
import titanicData from '../assets/titanic.csv';
```

Omdat er nu meerdere functies NA elkaar uitgevoerd moeten worden, maken we binnen `useEffect` functies aan. 

```js
React.useEffect(() => {
    const loadCSV = () => {
        readRemoteFile(titanicData, {
        download: true,
        header:true,
        dynamictyping: true,
        complete: (results) => {
                csvLoaded(results.data)
            }
        })
    }

    const csvLoaded = (data) => {
        console.log("DATA LOADED")
        console.log(data)
    }

    loadCSV()
}, [])
```
<br><br><br>

# Decision Tree

We gebruiken de Decision Tree JS code uit de PRG8 les, dit bestand kan je uit de les kopiëren naar je React Native project. We gaan voorspellen of Titanic passagiers de vakantie overleefd hebben, gebaseerd op hun geslacht, leeftijd en het aantal broers/zussen. We tekenen de tree niet, omdat de visualisatie library uit PRG8 niet werkt in React Native. 

```js
import { DecisionTree } from './decisiontree'
```
Nu kan je CSV data gebruiken om te trainen. Het model sla je op met `useRef` zodat het beschikbaar blijft in het component.

```js
const csvLoaded = (data) => {
    console.log("DATA LOADED")
    console.log(data[0])
    console.log(data[1])

    machine.current = new DecisionTree({
        ignoredAttributes: ["PassengerId", "Pclass", "Name", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
        trainingSet: results.data,
        categoryAttr: 'Survived'
    })
}
```
Vanaf dit punt kan je voorspellingen doen! 

```js
const makePrediction = () => {
    let pred = machine.current.predict({ Sex: "male", Age: 32, SibSp: 0 })
    let survived = (pred == 0 || pred == "0") ? "DECEASED" : "SURVIVED"
    setPrediction(survived)
}
```
Je kan de waarden uit een `state` halen die bij een invulveld hoort, zie hiervoor het bovenstaande KNN voorbeeld, en het uitgewerkte voorbeeld op Snacks.

- [Uitgewerkt Online Voorbeeld](https://snack.expo.dev/@eerk/decisiontree)

<br><br><br>

# Neural Networks

Omdat ML5 en BRAIN.JS niet werken in React Native, kan je [TensorFlowJS](https://www.tensorflow.org/js) gebruiken. 

```bash
npm install @tensorflow/tfjs
import * as tf from '@tensorflow/tfjs'
```

PSEUDOCODE TENSORFLOW
```js
// twee autos met horsepower, weight versus mpg
const inputTensor = tf.tensor2d([[30,3000],[12,1800]])
const labelTensor = tf.tensor1d([10,22])

// normalize
const normalizedInputs = ...
const normalizedLabels = ...

// bouw het neural network
const model = tf.sequential()
model.add(...)         // add layer one 
model.add(...)         // add layer two 
model.compile(...)     // bouw het model

// train het neural network
await model.fit(normalizedInputs, normalizedLabels, { epochs: 60} )
```
PSEUDOCODE PREDICTION
```js
const userInput = tf.tensor2d([[20,2000]]) // auto met 20 horsepower, 2000 kg
const normalizedUserInput = ...            // normalize de input
const normalizedPrediction = model.predict(normalizedUserInput) 
const actualPrediction = ...               // denormalize de prediction
```
- [Bekijk hier het volledige code voorbeeld](./tensorflowjs.md)
- [Bekijk hier een werkend voorbeeld in Expo Snacks](https://snack.expo.dev/@eerk/tensorflow-neural-network)


<br><br><br>

# Image Prediction

Dit kan je doen met de Tensorflow voor React library.

```bash
npm install @tensorflow/tfjs-react-native
```


Hieronder vind je een aantal voorbeelden:

- [Een foto voorspellen](https://blog.tensorflow.org/2020/02/tensorflowjs-for-react-native-is-here.html)
- [Een bodypose voorspellen](https://www.tensorflow.org/js/tutorials/applications/react_native)
- [Codesandbox TensorFlow examples](https://codesandbox.io/examples/package/@tensorflow/tfjs-react-native)

<br><br><br>

# ChatGPT Assistent

Het aanroepen van ChatGPT werkt hetzelfde als in de browser, omdat je `fetch` kan gebruiken:

```js
const callChatGPT = () => {
    const sentence = "how can technology help to achieve sustainability goals?"
    
    const bearer = 'Bearer ' + 'JOUW_API_KEY_HIER'
    const url = 'https://api.openai.com/v1/chat/completions'

    fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': bearer
        },
        body: JSON.stringify({
            model: 'gpt-3.5-turbo',
            messages: [
                {
                    role: 'user',
                    content: sentence
                }
            ],
            temperature: 0.7
        })
    })
        .then(response => response.json())
        .then(output => {
            const explanation = output.choices[0].message.content
            console.log(explanation)
        })
        .catch(error => console.log(error))
}
```