# Emotion Detection from Text (PyTorch)

##  Overview

This project focuses on building an emotion classification system from scratch using PyTorch, without relying on pretrained models. The goal was to understand how deep learning models learn language patterns and context by implementing custom pipelines and architectures.

---

##  Features

* Custom text preprocessing (tokenization, vocabulary building, padding)
* Supports large-scale dataset (~900K samples)
* Implemented:

  * CNN for local feature extraction
  * BiLSTM for contextual understanding
* Batch training using PyTorch DataLoader
* Evaluation using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* Emotion prediction with probability distribution
* Model saving and reloading for reuse

---

## Model Architecture

### CNN Model

* Embedding Layer
* Multiple Conv2D filters (3, 4, 5 n-grams)
* Max pooling
* Fully connected layer

### LSTM Model

* Embedding Layer
* Bidirectional LSTM
* Dropout for regularization
* Fully connected output layer

---

##  Evaluation Metrics

The model performance is evaluated using:

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1 Score

---

##  Saving & Loading

The model is saved along with:

* Model weights
* Vocabulary
* Label mapping
* Configuration (embedding size, sequence length, etc.)

This allows full reproducibility without retraining.

---

##  Inference

Users can input custom text, and the model returns:

* Predicted emotion
* Probability distribution across all emotions

Example:

```
Input: "I feel really happy today"
Output:
happy: 72.3%
neutral: 15.1%
sad: 5.4%
...
```

---

##  Limitations

* Performance depends heavily on vocabulary coverage
* Struggles with rare or unseen words
* Limited semantic understanding compared to transformer models

---

##  Future Improvements

* Reduce vocabulary size to improve generalization
* Add attention mechanism to LSTM
* Improve handling of unknown words
* Explore hybrid CNN + LSTM architecture

---
## Future Work 
 * Detecting Features in Audios and Recommending the Most similar Audio (Music Recommendation System) 
##  Tech Stack

* Python
* PyTorch
* NumPy
* Scikit-learn

---

##  Key Learning Outcomes

* Built NLP models without pretrained embeddings
* Understood trade-offs between CNN and LSTM architectures
* Learned importance of preprocessing and vocabulary design
* Implemented scalable training and evaluation pipelines

---