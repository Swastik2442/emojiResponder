# Sentiment Analysis with Emojis

The visual representation of a Sentiment can enhance communication by providing a more intuitive and engaging means of conveying Emotions. However, conventional Sentiment Analysis approaches typically produce basic classifications, such as "positive" or "negative," which may fail to capture the full spectrum of the Emotional Nuance.

This project aims to develop a Sentiment Analysis tool that translates text-based Sentiments into corresponding Emoji representations. Although it still does not captures the full spectrum, but it is a fun attempt at creating such a tool.

Access the **Jupyter Notebook** used in this project on [Google Colab](https://colab.research.google.com/drive/1PZY7FjCkSmJSRa7NT3gOenfj8FjSRBCX).

## Model Architecture

Training Data is of the Form (Text, Emoji ID). Text Embedding are extracted from the [GloVe](https://nlp.stanford.edu/projects/glove/) Pre-trained Word Vector (*glove.6B.50d*) and for each input Text, the input example becomes the Embedding of each Word in the Text.

The Input is split into Train, Test & Validate Sets and is used to train a Neural Network. The Network outputs the predicted Emoji Probabilities and is tested against the given Emojis.

## Infrastructure

The model is trained on Google Colab. The trained model is downloaded from there and uploaded to AWS S3. The uploaded model is accessed by an AWS Lambda Function which is invoked using the AWS API Gateway.

A Webpage served from AWS S3 is used to display an interface to the User and interacts with the API Gateway to get the predicted Emoji according to the Text input by the User.
