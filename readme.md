# emoC - Emotion Classification from Text

A research project that trains deep learning models from scratch to classify emotions in text, without relying on any pretrained embeddings or language models.

## Background

The idea behind this project comes from something pretty personal. Depression has a way of isolating people, and sometimes that isolation reaches a point where someone is on the edge. Being able to understand what someone is feeling - really understand it - takes time, consistency, and presence. That is something a machine cannot replace, and it should never try to.

But there is a scenario where you have no one around, and you are close to doing something you cannot undo. That is the gap this project is trying to explore: not to replace human connection, but to serve as a last resort when no one else is there. Beyond that, the longer goal is for machines to understand emotion so they can better preserve memories, nudge you toward your goals, and adapt how they interact based on how you are feeling.

After text, the plan is to extend this to audio and visuals, and eventually generate responses that match a given emotional tone.

## Project Structure

```
emoC/
├── Dataset/
│   ├── emotion_sentimen_dataset.csv
│   ├── go_emotions_dataset.csv
│   └── MergeDatasets.ipynb
├── Dataset_old/
│   ├── emotion_sentimen_dataset.csv
│   ├── emotion_sentimen_dataset.xlsx
│   ├── OldDataset_Analysis.png
│   └── preProcessedInputs/
│       ├── dataset.joblib
│       ├── max_len.pkl
│       ├── unique_emotion_list.pkl
│       └── vocab.pkl
├── cnn/
│   ├── full_pipeline.pth
│   └── text_cnn_full.pth
├── LSTM/
│   ├── LSTM_MODEL.pth
│   └── LSTM_MODEL.zip
├── sequential/
│   ├── emotion_model.h5
│   └── emotion_model.keras
├── Transformer/
│   └── transformer.pth
├── dataAnalyser.ipynb
├── dataPreprocessing.ipynb
├── MLP_CNN_LSTM.ipynb
└── transformer.ipynb
```

## Dataset

The original dataset (`emotion_sentimen_dataset.csv`) was analyzed and found to have a heavy class imbalance - around 80% of the samples were labeled as neutral. That made it too shallow for the kind of deep sentiment understanding this project is after.

To fix this, the dataset was merged with Google's GoEmotions dataset (`go_emotions_dataset.csv`), which covers 28 emotion categories from Reddit comments. The merged dataset is significantly more balanced and covers a wider emotional range.

**Emotion categories covered (28 total):**

| | | | |
|---|---|---|---|
| admiration | amusement | anger | annoyance |
| approval | caring | confusion | curiosity |
| desire | disappointment | disapproval | disgust |
| embarrassment | excitement | fear | gratitude |
| grief | joy | love | nervousness |
| optimism | pride | realization | relief |
| remorse | sadness | surprise | neutral |

**Dataset stats (GoEmotions portion):**

| Emotion | Count |
|---|---|
| neutral | 55,298 |
| approval | 17,620 |
| admiration | 17,131 |
| annoyance | 13,618 |
| gratitude | 11,625 |
| disapproval | 11,424 |
| curiosity | 9,692 |
| amusement | 9,245 |
| realization | 8,785 |
| optimism | 8,715 |
| disappointment | 8,469 |
| love | 8,191 |
| anger | 8,084 |
| joy | 7,983 |
| confusion | 7,359 |
| sadness | 6,758 |
| caring | 5,999 |
| excitement | 5,629 |
| surprise | 5,514 |
| disgust | 5,301 |
| desire | 3,817 |
| fear | 3,197 |
| remorse | 2,525 |
| embarrassment | 2,476 |
| nervousness | 1,810 |
| pride | 1,302 |
| relief | 1,289 |
| grief | 673 |

**Vocabulary size (old dataset):** 75,299 unique tokens

## Preprocessing Pipeline

The preprocessing is handled in `dataPreprocessing.ipynb` and covers:

1. Load the CSV and extract the text column
2. Normalize text: lowercase, strip punctuation
3. Tokenize using NLTK's `word_tokenize`
4. Build a vocabulary dictionary mapping each token to an integer index (starting from index 2, with 0 reserved for `<PAD>` and 1 for `<UNK>`)
5. Convert each sentence into a fixed-length integer sequence (max length: 60 tokens)
   - Shorter sequences are padded with zeros
   - Longer sequences are truncated
6. Convert labels from one-hot encoding to class indices
7. Convert everything to PyTorch tensors
8. Save preprocessed data to disk using `joblib` to avoid reprocessing each run

## Models

### CNN (`MLP_CNN_LSTM.ipynb`, `cnn/`)

A text CNN architecture inspired by Kim (2014), adapted for multi-class emotion classification.

- Embedding layer
- Multiple parallel Conv2D filters capturing 3-gram, 4-gram, and 5-gram patterns
- Max-over-time pooling
- Fully connected output layer

Trained on the old dataset. Saved as `cnn/text_cnn_full.pth` (weights only) and `cnn/full_pipeline.pth` (full pipeline including vocab and label map).

### BiLSTM (`MLP_CNN_LSTM.ipynb`, `LSTM/`)

A bidirectional LSTM that reads each sentence in both directions to capture context from both sides.

- Embedding layer
- Bidirectional LSTM
- Dropout for regularization
- Fully connected output layer

The BiLSTM was identified as one of the two architectures (along with Transformer) capable of the kind of deeper contextual understanding this project is working toward.

### Sequential / Keras Model (`sequential/`)

An earlier version of the model built using Keras. Saved as both `.h5` and `.keras` formats.

### Transformer (`transformer.ipynb`, `Transformer/`)

A custom transformer encoder built from scratch using PyTorch's `nn.TransformerEncoder`.

Architecture:

- Token embedding layer
- Learned positional embedding (not sinusoidal)
- Transformer encoder layers (multi-head self-attention + feedforward)
- Mean pooling over the sequence
- Dropout
- Fully connected classification head

Key hyperparameters used:
- `embed_dim = 128`
- `hidden_dim = 256`
- `dropout = 0.3`
- Vocabulary and label mappings loaded from the preprocessed pickle files

Saved as `Transformer/transformer.pth`.

## Evaluation

All models are evaluated with the following metrics computed using scikit-learn:

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)

## Inference

Once a model is loaded (along with its vocabulary and label mappings), you can pass any text string and get back the predicted emotion along with the probability distribution across all 28 classes.

Example output:

```
Input: "I feel really happy today"
Output:
  joy:         72.3%
  neutral:     15.1%
  sadness:      5.4%
  ...
```

## Setup

**Requirements:**

- Python 3.14+
- PyTorch 2.11+ (with CUDA 12.8 support, or CPU-only)
- NumPy
- Pandas
- scikit-learn
- NLTK
- joblib
- Matplotlib (for analysis notebooks)

**Install dependencies:**

```bash
pip install nltk pandas numpy scikit-learn joblib matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Download NLTK tokenizer data:**

```python
import nltk
nltk.download('punkt_tab')
```

**Run the notebooks in this order:**

1. `Dataset/MergeDatasets.ipynb` - merge the two datasets
2. `dataAnalyser.ipynb` - explore class distributions
3. `dataPreprocessing.ipynb` - build vocab and preprocess
4. `MLP_CNN_LSTM.ipynb` - train CNN and BiLSTM models
5. `transformer.ipynb` - train the transformer model

## Limitations

- Performance depends on vocabulary coverage. Words not seen during training map to `<UNK>`.
- The models have no semantic understanding of word relationships (no pretrained embeddings like GloVe or Word2Vec were used).
- The current max sequence length of 60 tokens may cut off context in longer sentences.
- Class imbalance in the dataset still exists for rarer emotions like grief, relief, and pride.

## What is next

- Train larger BiLSTM and Transformer models on the full merged dataset
- Add an attention mechanism to the BiLSTM
- Experiment with GloVe or FastText embeddings to improve handling of unseen words
- Reduce vocabulary noise to improve generalization
- Extend to audio: detect emotion from speech signals
- Build a recommendation system that suggests audio (music) based on the detected emotion

## Tech Stack

- Python
- PyTorch
- Keras / TensorFlow (early experiments)
- NumPy
- Pandas
- scikit-learn
- NLTK
- Matplotlib
- Joblib
