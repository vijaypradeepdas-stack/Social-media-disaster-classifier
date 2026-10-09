# Disaster Tweet Classification: Project Summary

## Overview

This project builds a system that reads a tweet and decides whether it is about a **real disaster** or **not**. It is a binary classification problem:

- **1** means the tweet is about a real disaster
- **0** means the tweet is not about a real disaster

The overall workflow is:

**Tweet → Clean Text → TF-IDF → Model → Prediction → Evaluation**

Two models are built and compared: a traditional machine learning model (Logistic Regression) and a simple neural network.

---

## Step 1: Importing the Tools

The project uses three groups of tools:

- **pandas** for loading and handling the data
- **scikit-learn** for splitting data, converting text to numbers, building the Logistic Regression model, and measuring performance
- **TensorFlow/Keras** for building and training the neural network

---

## Step 2: Loading the Dataset

Two files are used:

- **Training file:** contains the tweets along with their correct labels (disaster or not)
- **Test file:** contains tweets without labels, which the final model must predict

The size of both datasets is checked and the first few rows are displayed to understand what the data looks like.

---

## Step 3: Cleaning the Tweets

Raw tweets are messy, so they are cleaned to make them easier for a model to learn from. The cleaning process:

- Converts all text to lowercase so "Fire" and "fire" are treated as the same word
- Removes web links
- Removes @mentions
- Removes numbers, punctuation, and special symbols
- Removes extra spaces

The same cleaning is applied to both the training and test tweets. An example tweet is shown before and after cleaning to confirm that it works.

---

## Step 4: Splitting the Data

The training data is divided into two parts:

- **80% for training**, which is what the model learns from
- **20% for validation**, which is used to check how well the model performs on tweets it has never seen

The split keeps the same proportion of disaster and non-disaster tweets in both parts (stratified split), and a fixed random seed makes the results reproducible.

---

## Step 5: Converting Text into Numbers (TF-IDF)

Machine learning models cannot work with words directly, so each tweet is converted into numbers using **TF-IDF** (Term Frequency, Inverse Document Frequency).

TF-IDF gives each word a score based on two ideas:

- Words that appear often in a tweet are more important to that tweet
- Words that appear in almost every tweet carry less meaning and get a lower score

Key settings used:

- Only the **5,000 most useful words** are kept
- Common English filler words (such as "the", "is", "and") are removed

The vocabulary is learned from the training tweets only, then applied to the validation and test tweets. This avoids leaking information from unseen data into training.

---

## Step 6: Model 1, Logistic Regression (Baseline)

Logistic Regression is a simple and well-known classification method. It serves as the **baseline**, a reference point against which the more complex neural network can be compared.

The model is trained on the TF-IDF features of the training tweets, then used to predict labels for the validation tweets. Its performance is measured using:

- **Accuracy:** the share of tweets classified correctly
- **Classification report:** precision, recall, and F1-score for each class

---

## Step 7: Model 2, Simple Neural Network

A small neural network is built with three parts:

1. **Input layer:** receives the 5,000 TF-IDF features of a tweet
2. **Hidden layer:** 64 neurons using the **ReLU** activation, which helps the network learn non-linear patterns
3. **Output layer:** a single neuron using the **Sigmoid** activation, which produces a value between 0 and 1 that can be read as the probability that the tweet is about a disaster

The network is set up with:

- **Adam optimizer**, which adjusts the network's weights efficiently during learning
- **Binary cross-entropy loss**, the standard way to measure error for yes/no problems
- **Accuracy** as the metric to track

It is then trained for **5 epochs** (5 full passes through the training data) in batches of 32 tweets, while the validation data is monitored after each epoch.

---

## Step 8: Evaluating the Neural Network

After training, the network predicts a probability for each validation tweet. Any probability of **0.5 or higher** is treated as a disaster tweet, and anything lower as not a disaster. The predictions are then compared to the true labels using the same accuracy score and classification report used for Logistic Regression.

---

## Step 9: Comparing the Models

The accuracy of Logistic Regression and the neural network is placed side by side in a small table. This shows whether the added complexity of a neural network improves results over the simple baseline.

---

## Step 10: Predicting the Test Tweets

The neural network is used to classify every tweet in the test file. Using the same 0.5 threshold, each tweet is labelled as disaster or not. The results are saved in a **submission file** with two columns:

- **id:** the tweet's identifier
- **target:** the predicted label (0 or 1)

---

## Conclusion

The project shows a complete text classification workflow, from raw tweets to final predictions. TF-IDF turns text into numbers, Logistic Regression provides a simple baseline, and the neural network learns patterns from those numerical features.

### Main Concepts Demonstrated

- Text preprocessing
- TF-IDF feature extraction
- Binary classification
- Neural networks
- ReLU and Sigmoid activation functions
- Model training
- Model evaluation (accuracy, precision, recall, F1-score)
- Generating final predictions

> **Note:** The notebook as provided contains no saved output, so this summary describes what was done at each step rather than the specific accuracy scores obtained.
