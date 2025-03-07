# Emoji Responder

The visual representation of a Sentiment can enhance communication by providing a more intuitive and engaging means of conveying Emotions. However, conventional Sentiment Analysis approaches typically produce basic classifications, such as "positive" or "negative," which may fail to capture the full spectrum of the Emotional Nuance.

This project aims to develop a Sentiment Analysis tool that translates text-based Sentiments into corresponding Emoji representations. Although it still does not captures the full spectrum, but it is a fun attempt at creating such a tool.

Access the **Jupyter Notebook** used in this project on [Google Colab](https://colab.research.google.com/drive/1PZY7FjCkSmJSRa7NT3gOenfj8FjSRBCX).

## Model Architecture

Training Data is of the Form (Text, Emoji ID). Text Embedding are extracted from the [GloVe](https://nlp.stanford.edu/projects/glove/) Pre-trained Word Vector (*glove.6B.50d*) and for each input Text, the input example becomes the Embedding of each Word in the Text.

The Input is split into Train, Test & Validate Sets and is used to train a Neural Network with Keras. The Network outputs the predicted Emojis' Probabilities and the Emoji with the best Probability is checked against the actual Emoji.

## Infrastructure

The model is trained on Google Colab. The trained model is uploaded to an AWS S3 Bucket. An AWS Lambda Function is setup using a Docker Container Image, pushed on a repository in AWS ECR. It is used to access the uploaded model and predict an Emoji based on the Request made using its Function URL.

A Webpage served using AWS CloudFront from an AWS S3 Bucket is used to display an interface to the User. It sends requests to the Function URL to get the predicted Emoji according to the Text Input by the User.
