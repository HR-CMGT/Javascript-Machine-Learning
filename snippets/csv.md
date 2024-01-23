# Data laden

- CSV files
- JSON files
- Data filteren
- Training data en test data

<br>
<br>
<br>

## CSV files

Voeg de Papa Parse library toe aan je HTML

```HTML
<script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
```
of als je met modules werkt:
```bash
npm install papaparse
import Papa from './PapaParse/papaparse';
```

## Javascript

Download eerst een CSV file, bijvoorbeeld de [overlevenden van de Titanic](https://www.kaggle.com/c/titanic/data?select=train.csv). 

```javascript
function loadData(){
    Papa.parse("./data/titanic_survivors.csv", {
        download:true,
        header:true, 
        dynamicTyping:true,
        complete: results => checkData(results.data)
    })
}

function checkData(data) {
    console.table(data)
}
```
[Papa Parse Documentatie](https://www.papaparse.com)

<br>
<br>
<br>

## JSON files

Data in JSON formaat kan je ophalen met `fetch`. Hier gebruiken we `carsData.json` van google. 

```javascript
async function getData() {
    const response = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
    const json = await response.json()
    checkData(json)
}

function checkData(data) {
    console.log(data)
}
```

<br>
<br>
<br>

## Data filteren

Met [map](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map) kan je specifieke kolommen uit een dataset halen.

```javascript
const selectedColumns = data.map(car => ({
   mpg: car.mpg,
   horsepower: car.horsepower,
}))
```

Met [filter](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter) kan je checken of er geen ongeldige waarden in de data staan. Hier staan twee voorbeelden van filteren op `null` of op waarden die geen `number` zijn.

```javascript   
// check of waarden niet leeg zijn
const cleanData = data.filter(car => (
    car.mpg != null && 
    car.horsepower != null
))

// check of waarden een nummer zijn
const cleanData = data.filter(car => (
    !isNaN(car.mpg) && 
    !isNaN(car.horsepower)
))
```
Je kan `map()` en `filter()` aan elkaar plakken, dat ziet er bv. zo uit:

```javascript
const cleanData = data
    .map(day => ({
        MinTemp: day.MinTemp,
        MaxTemp: day.MaxTemp
    }))
    .filter(day => 
        typeof day.MinTemp === "number" &&
        typeof day.MaxTemp === "number"
    )
```

<br>
<br>
<br>

## Training en test data

Met `slice` kunnen we data opsplitsen in trainingdata en testdata. In dit geval is 80% van de data trainingdata en 20% is testdata.

```javascript
let trainData = data.slice(0, Math.floor(data.length * 0.8))
let testData = data.slice(Math.floor(data.length * 0.8) + 1)
```

> ⚠️ Om te voorkomen dat je data *gesorteerd* is op bv. het label moet je je array shufflen **voordat** je splitst op traindata en testdata.

```javascript
function shuffleArray(arr) {
    arr.sort(() => (Math.random() - 0.5))
}
```

<br>
<br>
<br>

## Links

- [Array map](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/map) 
- [Array filter](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter)
