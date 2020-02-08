# K-Nearest-Neighbour

Algorithm Basics. 

## Running the example

It's recommended to open `index.html` in a `localhost` server, for example by using the `live server` plugin in VS Code.

### No editor?

You can also create this project on `https://codesandbox.io`. Copy>paste the example code and the knear file, or fork the example from https://codesandbox.io/s/k-nearest-neighbour-example-k94dh

## Load the two JS files

```html
<script src="knear.js"></script>
<script src="app.js"></script>
```
## Instantiate kNear algorithm

```javascript
const k = 3
const machine = new kNear(k)
```

## Train the machine

Use data from this table to train the machine

| Hair color | Body length | Height | Weight | Ear length | Claws | Label |
| ---------- | ----------- | ------ | ------ | ---------- | ----- | ----- |
| black | 18 | 9.2 | 8.1 | 2 | true | 'cat' |
| grey | 20.1 | 17 | 15.5 | 5 | false | 'dog' |
| orange | 17 | 9.1 | 9 | 1.95 | true | 'cat' |
| brown | 23.5 | 20 | 20 | 6.2 | false | 'dog' |
| white | 16 | 9.0 | 10 | 2.1 | true | 'cat' |
| black | 21 | 16.7 | 16 | 3.3 | false | 'dog' |

### Code example

```javascript
machine.learn([10, 10, 10], 'cat')
```
## Classify new data

Test if the algorithm works with some fictional new data

```javascript
let catdog = machine.classify([7,7,7])
```

## Github documentation

https://github.com/NathanEpstein/KNear