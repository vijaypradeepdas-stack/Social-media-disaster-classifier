# Disaster Social Media Classification: Project Summary

## Overview

The notebook classifies text as disaster-related (`1`) or not disaster-related (`0`). It preserves the Twitter workflow and adds an experimental path for Instagram and Facebook-style captions.

## Data and limitation

- `train.csv` is the original labeled Twitter dataset and remains unchanged.
- `test.csv` contains unlabeled tweets used to create `submission.csv`.
- `social_posts_test.csv` contains 40 **synthetic**, labeled Instagram/Facebook-style examples. With seed `42`, the notebook holds out 10 examples (5 per class) and adds the other 30 (15 per class) to the tweet training split.

This means the training data is still dominated by tweets: the underlying labeled corpus is Twitter data, with only 30 synthetic social examples added in the current notebook run. Synthetic examples do not establish real-world cross-platform accuracy. Replace or add real, labeled Instagram/Facebook posts and keep a separate real-world holdout to measure performance on those platforms.

## Text processing and model

`clean_text()` lowercases text, removes URLs and mentions, separates camel-case hashtags, removes platform filler phrases, and normalizes repeated letters, punctuation, and emoji.

The notebook uses a stratified 80/20 split of the Twitter training data and a fixed random seed (`42`). TF-IDF is fit only on the combined training text, with up to 5,000 unigrams and bigrams. Its custom English stop-word list keeps `not`, `no`, `now`, and `under`.

Two classifiers are trained:

- Logistic Regression as a baseline.
- A TensorFlow/Keras neural network with 64 ReLU units and a sigmoid output, trained with Adam and binary cross-entropy for five epochs.

## Cross-platform evaluation

The neural network is evaluated on the Twitter validation split and the 10 held-out synthetic social posts. In the current seeded run:

| Evaluation set | Examples scored | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Twitter validation | 1,523 | 79.65% | 78.48% | 72.48% | 75.36% |
| Synthetic Instagram/Facebook holdout | 10 | 90.00% | 83.33% | 100.00% | 90.91% |

All 10 social holdout examples had known vocabulary and were scored. These results describe this small, synthetic test only; they must not be presented as real-world Instagram or Facebook accuracy.

## Using `predict_posts()`

Run the notebook through the cell that defines `predict_posts()`. Pass a pandas DataFrame containing a `text` column. A `platform` column is optional:

```python
posts = pd.DataFrame({
    "platform": ["Instagram", "Facebook"],
    "text": [
        "Evacuation notice issued after the wildfire moved closer to town.",
        "Sunday brunch with friends and fresh waffles.",
    ],
})

predictions = predict_posts(posts)
predictions[["platform", "text", "prediction", "disaster_probability"]]
```

The function returns the original columns plus `prediction` and `disaster_probability`. Predictions are `1` (disaster) or `0` (not disaster) when the cleaned post has known TF-IDF terms. If it has no known vocabulary, the prediction is `"uncertain"` and the probability is missing. The notebook's interactive section accepts one post at a time and exits when the user types `quit`.

## Running the project

Install the dependencies in the project virtual environment with `python -m pip install -r requirements.txt`, then open `disaster_social_media_handler.ipynb` in VS Code and select the project `.venv` kernel. Run its cells in order. The notebook predicts the unlabeled tweets in `test.csv` and saves them to `submission.csv`.
