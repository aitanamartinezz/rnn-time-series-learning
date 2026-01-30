# Sentiment Analysis on Amazon Reviews using Recurrent Neural Networks

This repository contains a deep learning project focused on **sentiment analysis**
using **Recurrent Neural Networks (RNNs)** applied to textual data. The objective
is to classify Amazon product reviews according to their sentiment by learning
temporal patterns in natural language sequences.

The project evaluates and compares different recurrent architectures, including
**SimpleRNN**, **LSTM**, and **GRU**, analyzing how model configuration and
hyperparameters impact classification performance.

---

## Problem Statement

Sentiment analysis is a common Natural Language Processing (NLP) task that involves
classifying text according to the expressed opinion or emotional tone. In this project,
the task is formulated as a **supervised text classification problem**, where the goal
is to predict the sentiment of a product review based on its textual content.

The main challenges addressed include:
- Modeling sequential dependencies in text
- Handling variable-length input sequences
- Understanding the impact of architectural and training choices on model performance

---

## Dataset Description

The dataset consists of **Amazon product reviews**, where each review is associated
with a numerical rating. Reviews are preprocessed and converted into labeled sentiment
classes:

- **Positive sentiment**: reviews with ratings of 4 or 5 stars  
- **Negative sentiment**: reviews with ratings of 1 or 2 stars  

The dataset is imbalanced, with a higher proportion of positive reviews. An initial
exploratory analysis is performed to study label distribution, vocabulary size, and
frequent terms.

---

## Repository Structure

 - sentiment_analysis_rnn.ipynb # Main notebook with full implementation
 - README.md # Project documentation
---

## Technologies

- Python
- Deep Learning
- Natural Language Processing (NLP)
- Recurrent Neural Networks (SimpleRNN, LSTM, GRU)
- Jupyter Notebook

---

## Data Preprocessing

The preprocessing pipeline includes:

- Text cleaning and tokenization
- Vocabulary construction and frequency analysis
- Integer encoding of words
- Sequence padding and truncation
- Preparation of input-label pairs
- Splitting data into training and validation sets

Sequence length is treated as a tunable hyperparameter to analyze its effect on
model performance.

---

## Model Architecture

Several recurrent neural network architectures are implemented and evaluated:

- **SimpleRNN**
- **LSTM**
- **GRU**

The models include:
- Embedding layers for word representation
- Stacked recurrent layers (2-layer and 4-layer configurations)
- Dense output layers with softmax activation for classification

Architectural depth and recurrent unit type are analyzed to understand their impact
on learning sentiment patterns.

---

## Experimental Setup

A series of experiments is conducted to evaluate the influence of different
hyperparameters. In each experiment, only one parameter is varied while others
are kept fixed.

The following hyperparameters are explored:

- **Sequence length** (e.g. 50, 100, 150, 300)
- **Embedding size** (e.g. 100, 500, 1000, 5000)
- **Batch size** (e.g. 32, 128, 256)
- **Model type** (SimpleRNN, LSTM, GRU)
- **Network depth** (small vs medium configurations)

---

## Results

### Overall Performance

Models achieve accuracies above **84%**, with several configurations exceeding **86%**.

### Key Findings

- **LSTM and GRU consistently outperform SimpleRNN**
- **LSTM models achieve the best overall performance**
- Longer sequence lengths (especially 150 tokens) lead to better results
- Larger embedding sizes improve performance by capturing richer semantic information
- Larger batch sizes generally yield higher accuracy, though smaller batches remain competitive

### Best Performing Model

- **LSTM_Medium_seq150**
- Accuracy: **87.02%**

This configuration demonstrates the importance of both model capacity and sufficient
context length in sentiment classification tasks.

---

## Analysis and Observations

- Bidirectional architectures are not required to achieve strong performance in this task
- LSTM models handle long-term dependencies more effectively than SimpleRNN
- GRU models provide competitive results with fewer parameters
- Very short sequences limit contextual understanding and reduce accuracy

---

## Conclusion

This project demonstrates the effectiveness of recurrent neural networks for
sentiment analysis on real-world textual data. The experiments highlight how
architectural choices and hyperparameters influence model performance in NLP tasks.

Among the evaluated architectures, **LSTM models provide the best balance between
performance and stability**, making them well-suited for sentiment classification
on large-scale review datasets.

The implementation provides a strong foundation for future extensions, such as:
- Bidirectional recurrent models
- Attention mechanisms
- Transformer-based architectures

---

## Author

Aitana Martínez
