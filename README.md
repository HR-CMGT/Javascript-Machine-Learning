# Javascript Machine Learning

This repository maintains a reading list of Machine Learning technologies, algorithms and API's focused on javascript. 

### Why Javascript?

- Can run on any web client, the end user doesn't have to install anything.
- Easy integration with device sensors such as camera, microphone, gyroscope, light sensor, etc.
- End user does not have to send privacy-sensitive data to a server.
- Easy integration with user interface and interactive elements.
- Easy install process for developer, less sensitive to computer setup issues, python setup issues, version conflicts.
- Convert your AI code to a web server with one line of code.
- Can still use models trained in Python.
- Training uses the same underlying C / GPU capabilities of your computer.

> *Python is used in the science community, so there are much more python libraries for math and computer science subjects available. Most AI tutorials will use python.*

<Br>

# Contents

- [Getting started](#gettingstarted)
- [Neural networks](#neuralnetworks)
- [Algorithms](#algorithms)
- [Language models](#languagemodels)
- [Image recognition](#images)
- [Speech](#speech)
- [Loading and evaluating data](#loadingdata)
- [Datasets](#datasets)
- [Community](#community)
- [Cloud GPU](#cloud)
- [Random links](#examples)

<br>
<br>
<br>

![dog](./images/nn.png)

## <a name="gettingstarted"></a>Getting started with Machine Learning

- [🔥 Elements of AI part 1](https://course.elementsofai.com) and [part 2](https://buildingai.elementsofai.com)
- [De Nederlandse AI Cursus](https://www.ai-cursus.nl)
- [Brilliant.org - tutorial app with daily questions](https://brilliant.org/courses/artificial-neural-networks/)
- [A visual introduction to Machine Learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)

<br>

## <a name="neuralnetworks"></a>Neural Networks in Javascript

- [ML5](https://learn.ml5js.org/#/reference/neural-network) 
- [BrainJS](https://brain.js.org/#/getting-started)
- [TensorflowJS](https://www.tensorflow.org/js).
- [Build a Neural Network in JS without any libraries](https://dev.to/grahamthedev/a-noob-learns-ai-my-first-neural-networkin-vanilla-jswith-no-libraries-1f92)
- [Get started with ML5 neural networks](https://learn.ml5js.org/#/reference/neural-network) and add [hidden layers](./snippets/layers.md)
- [Building your first Neural Network in Tensorflow JS](https://towardsdatascience.com/build-a-simple-neural-network-with-tensorflow-js-d434a30fcb8)
- [Regression with Tensorflow](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html)
- [Getting started with TensorflowJS](https://curiousily.com/posts/getting-started-with-tensorflow-js/)
- [CodePen](https://codepen.io/topic/tensorflow/templates), [Traversy](https://youtu.be/tZt6gRlRcgk) and [W3Schools](https://www.w3schools.com/ai/ai_tensorflow_intro.asp) tensorflow tutorials.
- [Build rock-paper-scissors with Reinforcement Learning](https://towardsdatascience.com/a-beginners-guide-to-reinforcement-learning-using-rock-paper-scissors-and-tensorflow-js-37d42b6197b5) and [github](https://github.com/sachag678/freeCodeCamp)
- [Reinforcement Learning library](https://epic-darwin-f8b517.netlify.app) with [github](https://github.com/ttumiel/gym.js/) and [game tutorial](https://medium.com/@pierrerouhard/reinforcement-learning-in-the-browser-an-introduction-to-tensorflow-js-9a02b143c099)

<br>

## <a name="algorithms"></a>Algorithms in Javascript

- [A visual tour of machine learning algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)
- [Machine Learning algorithms for newbies](https://towardsdatascience.com/a-tour-of-the-top-10-algorithms-for-machine-learning-newbies-dde4edffae11)
- [Math concepts for Programmers](https://www.youtube.com/watch?v=2SpuBqvNjHI)
- [K-Nearest Neighbour](https://github.com/NathanEpstein/KNear) and an [explanation of writing your own KNN in Javascript](https://burakkanber.com/blog/machine-learning-in-js-k-nearest-neighbor-part-1/)
- [K-Means](https://miguelmota.com/blog/k-means-clustering-in-javascript/)
- [Decision Tree, Random Forest](https://github.com/lagodiuk/decision-tree-js) and [Regression Tree](https://winkjs.org/wink-regression-tree/)
- [Movie Recommender System in Javascript](https://github.com/javascript-machine-learning/movielens-recommender-system-javascript) and a [quick and dirty tutorial on building your own recommender system](https://dev.to/jimatjibba/build-a-content-based-recommendation-engine-in-js-2lpi)
- [Create a self-driving car with a evolving genetic algorithm](https://github.com/gniziemazity/Self-driving-car) and another example on [dev.to](https://dev.to/trekhleb/self-parking-car-in-500-lines-of-code-58ea)

<br>

## <a name="languagemodels"></a>Language models and API's for Javascript

- [LangChain LLM library](https://js.langchain.com/docs/get_started/introduction)
- [OpenAI ChatGPT API](https://platform.openai.com/docs/introduction)
- [LLama API](https://www.llama-api.com), [Mistral API](https://docs.mistral.ai), [Claude API](https://support.anthropic.com/en/collections/5370014-claude-api)
- Use [OLLama](https://ollama.ai) to run language models locally, talk to the LLM with the built-in webserver.
- [Geitje](https://goingdutch.ai/en/posts/introducing-geitje/) is a Dutch large language model.
- [Basics of language processing](https://towardsdatascience.com/natural-language-processing-from-basics-to-using-rnn-and-lstm-ef6779e4ae66), [Natural Language Processing in Javascript](http://naturalnode.github.io/natural/), [Paperspace tutorial](https://blog.paperspace.com/training-an-lstm-and-using-the-model-in-ml5-js/)
- [What are word vectors?](https://gist.github.com/aparrish/2f562e3737544cf29aaf1af30362f469) and [📺  Understanding Word2Vec](https://youtu.be/MOo9iJ8RYWM)

<br>

## <a name="images"></a>Image recognition

- [Google MediaPipe for pose, object, image, segment detection](https://mediapipe-studio.webapps.google.com/home) 
- [ML5 Image Classifier](https://learn.ml5js.org/#/reference/image-classifier) and [Object Detection](https://learn.ml5js.org/#/reference/object-detector) in ML5.js
- [Create teachable machine from scratch with TensorFlowJS](https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine#16)
- [Use KNN to classify poses with ML5](https://learn.ml5js.org/#/reference/knn-classifier) 
- [Recognise handwriting in Javascript](https://github.com/cazala/mnist)
- [Face-JS, a library to track facial expressions](https://justadudewhohacks.github.io/face-api.js/docs/index.html)
- [Hand Tracking JS](https://victordibia.github.io/handtrack.js/)
- [Image recognition with your own images](./snippets/extractfeatures) and [Feature Extraction documentation](https://ml5js.org/reference/api-FeatureExtractor/)

<br>

## <a name="speech"></a>Speech

- [Generate voices with Elevenlabs](https://elevenlabs.io)
- [Listen to human speech in the browser](https://developer.mozilla.org/en-US/docs/Web/API/Web_Speech_API/Using_the_Web_Speech_API), and a [Simple demo of listening and speaking with javascript](https://github.com/mdn/web-speech-api)
- [OpenAI Whisper](https://platform.openai.com/docs/guides/speech-to-text)
- [Recognise sound using the browser, microphone, and TensorflowJS](https://dev.to/devdevcharlie/acoustic-activity-recognition-in-javascript-2go4)
- [Code example web speech](./snippets/speech.md) and [webcam](./snippets/camera.md)

<br>

## <a name="loadingdata"></a>Loading and evaluating data

- [Load CSV data in javascript with Papa Parse](https://www.papaparse.com), and [code example to load and filter data](./snippets/csv.md)
- Visualise data with [D3.js](https://d3js.org), [VEGA](https://vega.github.io/vega/), [ChartJS](https://www.chartjs.org) or [PlotlyJS](https://plotly.com/javascript/)
- [Code example for drawing Scatterplot from a CSV file](./snippets/scatterplot.md)
- [Code example for data normalisation](./snippets/normalise.md)
- [Visualise Tensorflow with TFVis](https://js.tensorflow.org/api_vis/1.4.3/)
- [Manipulate large amounts of data with Danfo.js](https://danfo.jsdata.org)

<br>

## <a name="datasets"></a>Datasets

- [Google Dataset Search](https://datasetsearch.research.google.com) 
- [UCI Machine Learning Repository](https://archive-beta.ics.uci.edu)
- [Socrata Open Data search](https://dev.socrata.com/data/)
- [Kaggle Datasets](https://www.kaggle.com/datasets/)
  - [Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
  - [Mobile Phone Prices](https://www.kaggle.com/datasets/pratikgarai/mobile-phone-specifications-and-prices)
  - [UFO sightings](https://www.kaggle.com/NUFORC/ufo-sightings)
  - [Football results](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017)
  - [Speed Dating](https://www.kaggle.com/datasets/annavictoria/speed-dating-experiment) and [Marriage](https://www.kaggle.com/aagghh/divorcemarriage-dataset-with-birth-dates)
  - [Spam detection](https://www.kaggle.com/datasets/veleon/ham-and-spam-dataset)
- [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html)
- [MovieLens - data on 45.000 movies by 270.000 users](https://www.kaggle.com/rounakbanik/the-movies-dataset/data)
- [🐱 440 Cat meows in different contexts](https://zenodo.org/record/4008297#.YNGgey8RppQ) and [Cats per square kilometer](https://data.gov.uk/dataset/9d475e06-3885-4a90-b8c0-77fea13f92e6/cats-per-square-kilometre)
- [Speech Audio dataset](https://keithito.com/LJ-Speech-Dataset/)
- [QuickDraw - Doodles dataset](https://github.com/googlecreativelab/quickdraw-dataset/)
- [COCO - Common Objects in Context](https://cocodataset.org/#home)
- [Rotterdam Open Data](http://rotterdamopendata.nl/dataset) en [Rotterdam 3D data](https://www.3drotterdam.nl/#/)
- [Netherlands Open OV Data](https://www.openov.nl)
- [Traffic data "persons in urban traffic scenes" TU Delft](https://eurocity-dataset.tudelft.nl)
- [Dataregister van de Nederlandse Overheid](https://data.overheid.nl)
- [Cars data with 20.000 images of 190 types of car](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Celebrity faces dataset on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset) and [source](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- [International Football Results from 1872 to 2017](https://www.kaggle.com/martj42/international-football-results-from-1872-to-2017)
- [Luchtmeetnet luchtvervuiling CSV data](https://data.rivm.nl/data/luchtmeetnet/Actueel-jaar/)
- [Daily stock prices](https://www.alphavantage.co/documentation/)

<br>

## <a name="community"></a>Community

- [AI Stackoverflow](https://ai.stackexchange.com)
- [Kaggle - Machine Learning challenges](https://www.kaggle.com)
- [Welcome.ai Instagram](https://www.instagram.com/welcome.ai/)
- [AI x Design Community](https://www.aixdesign.co), [Instagram](https://www.instagram.com/aixdesign.community/) and [Resources Reading List](http://bit.ly/aixd-library)
- [Rotterdam AI Meetup](https://www.meetup.com/data-science-rdm/events/)

<br>

## <a name="cloud"></a>Cloud GPU

- [Huggingface Spaces](https://huggingface.co/spaces)
- [Google Vertex AI](https://cloud.google.com/vertex-ai)
- [Google Colab](https://colab.research.google.com)
- [PaperSpace](https://blog.paperspace.com/)
- [Lambda Labs](https://lambdalabs.com/service/gpu-cloud)


<br>

## <a name="examples"></a>Random links

A collection of interesting links, tools and tutorials

<a href="https://playground.tensorflow.org" target="_blank">![playground](./images/playground.png)</a>

- [Quick AI Introduction with javascript code examples](./snippets/introduction.md)
- [Generate images with the Dall-e API](https://openai.com/blog/dall-e-api-now-available-in-public-beta)
- [Tensorflow Playground](https://playground.tensorflow.org) 
- [React Native AI code snippets](./snippets/reactnative/)
- [Visualise how a Neural Network recognises numbers](https://www.cs.cmu.edu/~aharley/vis/conv/flat.html)
- [📺 Build a security camera with TensorflowJS and React](https://www.youtube.com/watch?v=7QBYX65t7Mw)
- [Integrating TensorflowJS into your User Interface with Parcel](https://medium.com/codingthesmartway-com-blog/tensorflow-js-crash-course-machine-learning-for-the-web-getting-started-50694a575238)
- [📺 Coding a single neuron (perceptron) in Javascript](https://youtu.be/o98qlvrcqiU) and the [result](https://beta.observablehq.com/@mpj/neural-network-from-scratch-part-1)
- [Use BrainJS to get started with a simple neural network](https://brain.js.org/#/getting-started) or watch the [📺 video tutorial](https://www.youtube.com/watch?v=RVMHhtTqUxc)
- [Perceptron (single neuron) code snippet](https://gist.github.com/primaryobjects/dfb8927f9f0ca21b6a24647168cead41)
- [Creating a Recommender System in Javascript](https://github.com/javascript-machine-learning/movielens-recommender-system-javascript)
- [😱 hacking a model to make a wrong prediction with Adversarial.js](https://kennysong.github.io/adversarial.js/)
- [Predict your location in your home using the strength of your wifi signal](https://dev.to/devdevcharlie/predicting-indoor-location-using-machine-learning-and-wifi-information-m78)
- [Using a Javascript Perceptron to classify dangerous snakes](https://github.com/elyx0/rosenblattperceptronjs)
- [Classify an image on Codepen in 5 lines of Javascript](https://codepen.io/eerk/pen/JmKQLw)
- [Neural Drum Machine](https://codepen.io/teropa/pen/JLjXGK) and [Voice-based beatbox](https://codepen.io/naotokui/pen/NBzJMW) created with [MagentaJS](https://magenta.tensorflow.org)
- [Evolving Genetic Algorithm with Flappy Bird](https://github.com/ssusnic/Machine-Learning-Flappy-Bird)
- [Google AI experiments](https://experiments.withgoogle.com/ai)
- [Watch a perceptron learn](https://kokodoko.github.io/perceptron/)
- [Synaptic JS Neural Networks](http://caza.la/synaptic/) and [Tutorial](https://medium.freecodecamp.org/how-to-create-a-neural-network-in-javascript-in-only-30-lines-of-code-343dafc50d49)
- [Control a ThreeJS game with Teachable Machine](https://github.com/charliegerard/whoosh)
- [Silence of the Fans - running TensorflowJS on Google Colab](https://dev.to/obenjiro/silence-of-the-fans-part-1-javascript-quickstart-5f3m) - and [code example](bit.ly/colabjs)
- [Using AutoEncoders with TensorflowJS tutorial](https://douglasduhaime.com/posts/visualizing-latent-spaces.html)
- [Figment uses an intuitive UI to use AI tools without coding](https://figmentapp.com)
- [Hogeschool Rotterdam Datalab: Rob van der Willigen's Blog about AI](https://robfvdw.medium.com)
- Google API's for [Translate](http://cloud.google.com/translate/), [Vision](http://cloud.google.com/vision/), [Speech](http://cloud.google.com/speech/) and [Language Processing](http://cloud.google.com/natural-language/)
- [Microsoft Azure Machine Learning APIs](https://gallery.azure.ai/machineLearningAPIs)
- [Teachable Machine](https://teachablemachine.withgoogle.com) and [Lobe.AI](https://lobe.ai) can export a model and all necessary code for your website.
- [Cognimates: An AI education platform for building games, programming robots & training AI models](http://cognimates.me/home/)
- [Obviously.ai](https://www.obviously.ai) can train a model with CSV data right in the browser.
- Check the [Hardware, Python and Games](./snippets/python.md) reading list.

### Disclaimer

Since the field of AI is evolving so rapidly, some links may be outdated. This list is maintained by [Erik Katerborg](https://kokodoko.github.io).

>*Last updated: 22 january 2024.*

<br>

![SelfDrivingCar](https://imgs.xkcd.com/comics/self_driving_car_milestones.png)

[https://xkcd.com/1925/](https://xkcd.com/1925/)  