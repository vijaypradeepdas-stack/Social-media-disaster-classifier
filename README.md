# Social-media-disaster-classifier

A binary text classification project that reads a tweet and predicts whether it is about a **real disaster** or **not**.

- **1** → tweet is about a real disaster
- **0** → tweet is not about a real disaster

The full workflow:


Two models are built and compared:
- **Logistic Regression** (baseline)
- **Simple Neural Network** (TensorFlow/Keras)

---

## 📁 Project Structure


---

## 🧰 Tools Used

- **pandas** — loading and handling data
- **scikit-learn** — train/test split, TF-IDF, Logistic Regression, evaluation metrics
- **TensorFlow / Keras** — building and training the neural network

---

## 🚀 Pipeline

### 1. Load the Dataset
- **Training file:** tweets with correct labels
- **Test file:** tweets without labels, to be predicted

### 2. Clean the Tweets
Raw tweets are messy, so the following cleaning steps are applied to both training and test data:
- Convert text to lowercase
- Remove URLs
- Remove `@mentions`
- Remove numbers, punctuation, and special symbols
- Remove extra whitespace

### 3. Split the Data
- **80%** training
- **20%** validation
- Stratified split + fixed random seed for reproducibility

### 4. Convert Text to Numbers (TF-IDF)
Each tweet is converted into numerical features using **TF-IDF**:
- Keeps only the **5,000 most useful words**
- Removes common English stopwords
- Vocabulary learned **only from training data** to avoid leakage

### 5. Model 1 — Logistic Regression (Baseline)
A simple, well-known classifier used as a reference point.
Evaluated using:
- **Accuracy**
- **Classification report** (precision, recall, F1-score)

### 6. Model 2 — Simple Neural Network
| Layer | Details |
|-------|---------|
| Input | 5,000 TF-IDF features |
| Hidden | 64 neurons, **ReLU** activation |
| Output | 1 neuron, **Sigmoid** activation (probability of disaster) |

Training setup:
- **Optimizer:** Adam
- **Loss:** Binary cross-entropy
- **Metric:** Accuracy
- **Epochs:** 5
- **Batch size:** 32

### 7. Evaluate the Neural Network
- Predictions ≥ **0.5** → disaster (1)
- Predictions < **0.5** → not disaster (0)
- Compared against true labels using the same metrics as the baseline

### 8. Compare Models
Accuracies of both models are placed side by side to see whether the neural network improves over the baseline.

### 9. Predict the Test Set
The neural network labels every tweet in the test file and saves the results to **`submission.csv`**:

| Column | Description |
|--------|-------------|
| `id` | Tweet identifier |
| `target` | Predicted label (0 or 1) |

---

## 📊 Results

> Specific accuracy scores are not included in this summary. Run the notebook to generate them.

| Model | Accuracy |
|-------|----------|
| Logistic Regression | — |
| Neural Network | — |

---

## 🧠 Concepts Demonstrated

- Text preprocessing
- TF-IDF feature extraction
- Binary classification
- Neural networks
- ReLU and Sigmoid activation functions
- Model training
- Model evaluation (accuracy, precision, recall, F1-score)
- Generating final predictions

---

## 🛠️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO

   pip install pandas scikit-learn tensorflow

   jupyter notebook disaster_tweets.ipynb



---

### What to change before pushing

- Replace `YOUR_USERNAME/YOUR_REPO` with your actual repo path.
- Rename the notebook file if yours is different (I used `disaster_tweets.ipynb`).
- Add the accuracy numbers to the **Results** table once you run the notebook.
- Add a `LICENSE` file if you keep the license section, otherwise delete it.

Want me to also make a shorter version, or one with badges (Python version, TensorFlow, etc.) at the top?
