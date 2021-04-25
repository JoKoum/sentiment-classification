# MLP Sentiment classification

The project investigates the performance of the Multi-layer Perceptron algorithm, demonstrated at the [Sentiment Analysis lesson](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-analysis-network) from Udacity Deep Learning Nanodegree, when trained with the data from the artificially translated in Greek dataset of the [Machine Learning project](https://github.com/JoKoum/Machine-Learning-Project). Translated reviews are passed to the neural network represented in a simplified bag-of-words model.

<img src="static/sentiment_network_sparse_2.png" align="middle"/>

The trained neural network is used to predict sentiment of the reviews submitted by users at a web app inderface, using a modified version of the index.html from the ['Deploying a Sentiment Analysis Model' project](https://github.com/JoKoum/Udacity-Deep-Learning/tree/master/AWS) and Flask.