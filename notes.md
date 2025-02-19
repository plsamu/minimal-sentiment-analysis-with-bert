# raw notes (newest to the top)

## BertModel

<details>
  <summary>What is BERT? What is a BERTModel?</summary>
BERT (Bidirectional Encoder Representations from Transformers) is a deep learning model developed by Google in 2018 for Natural Language Processing (NLP) tasks. It is based on the Transformer architecture and is designed to understand the context of words in a sentence bidirectionally.<br><br>

Features:

- pretrained on large datasets
- can be fine-tuned
    - backpropagation (updates the weights)
    - loss function (cross-entropy for classification, MSE for regression, ...)
    - add classification layer, or regression layer, or other "header"

</details>


## Modify model based on user feedback

This could be done using a form of gradient-based learning or through a more specific adaptation method (e.g., fine-tuning, gradient updates, or adding a custom layer to modify outputs).

Backpropagation based on user feedback aka alter the behavior of the downstream model.

- fine-tuning
- gradient adjustment
- applying a scaling factor to the embeddings

## roadmap

1. Embedding + Hashing or Dimensionality Reduction
2. User feedback
3. Modify model based on user feedback

## backpropagation

| Method name                | Backpropagation                              |
| -------------------------- | -------------------------------------------- |
| Reinforcement learning     | Yes, indirectly (for deep RL)                |
| Online learning            | Yes (for neural networks in online settings) |
| Adaptive embeddings        | Yes                                          |
| Neural network fine-tuning | Yes                                          |
| Collaborative filtering    | Not typically                                |
| Gradient descent           | Yes, in the context of neural networks       |
| Bayesian inference         | No                                           |


## approach to tune the model 2

- Reinforcement learning: Adapt over time based on rewards/penalties.
- Online learning: Incrementally update the model with each feedback.
- Adaptive embeddings: Fine-tune embeddings in real-time with user input.
- Neural network fine-tuning: Adjust model parameters (weights) through backpropagation.
- Collaborative filtering: Use user feedback to adjust predictions in a recommendation system.
- Gradient descent: Adjust model parameters using loss functions based on feedback.
- Bayesian inference: Update predictions and model beliefs based on incoming feedback.

## approach to tune the model

- Fine-Tuning the Model with User Feedback (Supervised Learning)
- Active Learning: The Model Requests User Feedback
- Reinforcement Learning from User Feedback
    - "tanti snake e poi prendi quello che ha fatto meglio"

## milestone

- I'm looking for a continual learning or active learning approach

## legenda 1

- convert the text to a tensor : embedding

## generate

To generate an integer that represents the value of a sentence (e.g., "hello world"), one common approach is to use the output of a model (such as a transformer-based model like BERT, GPT, or other pre-trained models) and map the sentence to a numerical representation, such as an embedding or a classification output.

However, since you want a single integer to represent the sentence, I will outline two potential ways to accomplish this:

- Approach 1: Using Pre-trained Language Models (e.g., BERT) to Obtain Sentence Embeddings
   	- Embedding + Hashing or Dimensionality Reduction
   	- You can convert the sentence into an embedding and then reduce that embedding to a single integer by applying a hashing function or dimensionality reduction.

- Approach 2: Using a Classification Model to Directly Map Sentences to a Label
   	- You can train or use a pre-trained model that maps sentences directly to an integer (e.g., for text classification, sentiment analysis, etc.).

## tensor 1

Let’s say you have a tensor with shape (3, 5, 768):

- Batch size (3): There are 3 sequences in the batch.
- Sequence length (5): Each sequence has 5 tokens (padded if needed to align lengths).
- Hidden size (768): Each token is represented by a 768-dimensional feature vector.

For example:

tensor(3, 2, 10)

- 3 batches
- 2 token each batch
- 10 hidden_size for each token

```json
[
  [  # 1st sequence (Batch index 0)
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Token 1
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],  # Token 2
  ],
  [  # 2nd sequence (Batch index 1)
    [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],  # Token 1
    [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],  # Token 2
  ],
  [  # 3rd sequence (Batch index 2)
    [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],  # Token 1
    [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],  # Token 2
  ]
]
```

## types 1 - model output

- <https://huggingface.co/docs/transformers/en/main_classes/output>

## looking at the code 1

```txt
tokenized input:
    processed version of raw text, where the text is broken down into smaller units (called tokens)
    that can be understood by a machine learning model, typically a transformer like BERT, GPT, or similar models.
    Tokenized input also includes additional metadata, like attention masks, that help the model process the input correctly.

gradients:
    Are necessary for backpropagation, where these stored values are used to compute gradients for weight updates.
    When in inference, gradient computation is not needed because you're not updating the model's weights.
	- no_grad() prevents PyTorch from tracking and storing gradients during computations (useful when you're performing inference)

pt:
    Stands for PyTorch. return_tensors="pt" means that will return a PyTorch tensor
```

## search input 1

Model that dynamically adjusts based on user feedback and assigns a value to a block of text tailored to the user's interests, likes, and dislikes

1. Use semantic embeddings to represent the text in a numerical format. These embeddings will capture the meaning of the text.
2. Add User Preference Representation
   1. Maintain a user profile vector that reflects the user’s likes and dislikes

## Feature Extraction

1. Sentiment Analysis
   - Assign a polarity score (e.g., positive/negative).
   - Tools: TextBlob, VADER, or machine learning models.
   - Logic: Text can convey emotions, and capturing this sentiment helps determine its "tone."
2. Topic Modeling
   - Group text into topics using algorithms like LDA or pre-trained models (e.g., BERT).
   - Logic: Topics provide a sense of what the text is about.
3. Semantic Embeddings
   - Convert text into dense vectors using models like Word2Vec, GloVe, or Sentence-BERT.
   - Logic: Embeddings capture contextual meaning and relationships between words.
4. Keyword Analysis
   - Identify specific words or phrases related to your problem (e.g., "urgent," "critical").
   - Logic: Keywords provide targeted insights for specific applications.