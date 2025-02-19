from typing import List
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bert import BertModel, BertConfig
from transformers import AutoModel, BatchEncoding
import torch
from torch import FloatTensor, Tensor
import torch.nn as nn
from torch.types import Number
from torch.optim.adamw import AdamW
from torch.nn import MSELoss


class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone: BertModel = AutoModel.from_pretrained(MODEL_NAME)

        # Freeze backbone parameters (do not optimize them)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Define last layer
        # Get embedding size from model config
        embedding_size_config: BertConfig = self.backbone.config

        # create last custom more little layer
        self.last_layer = nn.Sequential(
            # get backbone model's output and transform it from hidden_size to size 1
            nn.Linear(embedding_size_config.hidden_size, 1),
            # then apply sigmoid function to scalar returning a value between 0 to 1
            nn.Sigmoid()
        )

    def forward(self, tokenized_inputs: BatchEncoding) -> FloatTensor:
        """
        The use of the sigmoid make the use of the attention_mask useless

        attention_mask = torch.as_tensor(tokenized_inputs["attention_mask"], dtype=torch.float32)
        """
        embeddings: BaseModelOutput = self.backbone(**tokenized_inputs)
        # batch_size == 1 mean only 1 text input
        assert embeddings.last_hidden_state.shape[0] == 1  # check batch_size
        # CLS = Classification Token
        # Get CLS token embedding, shape: (batch_size, CLS, hidden_size)
        cls_embedding = embeddings.last_hidden_state[:, 0, :]
        # apply last custom little layer (for better performance) to CLS tensor, the shape will be (batch_size, 1) aka (1, 1)
        sigmoid_tensor = self.last_layer(cls_embedding)
        return sigmoid_tensor


# Load a popular lightweight embedding pre-trained model and recover a tokenizer service based on the model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MY_MODEL = MyModel()

# Define optimizer (only optimize last_layer)
# AdamW decouples weight decay from the gradient update step (reduces overfitting)
# Define optimizer. Use a smaller that default learning rate.
# parameters: includes learnable weights and biases
# parameters() call will always return references to the same tensors so this object can be static
OPTIMIZER = AdamW(MY_MODEL.last_layer.parameters(), lr=1e-5)

# optimize
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL.to(DEVICE)


def make_prediction(texts: List[str]) -> Number:
    tokens: BatchEncoding = TOKENIZER(
        texts, padding=True, truncation=True, return_tensors="pt")
    sigmoid_tensor = generate_embeddings(tokens)
    return embeddings_to_scalar_sum(sigmoid_tensor)


def mean_pooling(outputs: BaseModelOutput, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling, considering attention mask.
    """
    token_embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size())

    # Apply the attention mask and sum embeddings over the sequence length
    # (batch_size, hidden_size)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(
        dim=1), min=1e-9)  # (batch_size, 1)

   # Mean pooling across multiple batch
    mean_pooled = sum_embeddings / sum_mask  # (batch_size, hidden_size)

    # Reduce across the batch dimension to produce a single vector
    final_vector = torch.mean(mean_pooled, dim=0)  # Shape: (hidden_size)

    return final_vector


def print_parameters():
    for name, param in MY_MODEL.named_parameters():
        if param.requires_grad:
            print(f"{name} before: {param.data.norm().item()}")


def compute_backpropagation(texts: List[str], user_feedback: float):
    """
    User provides feedback to influence model.

    The first experiment was to calculate target_tensor using multiplying actual_tensor * user_feedback
    but a friend said to me that for definition target_tensor is independent from model output (actual_tensor).
    So I started to use zeroes and ones as target_tensor with a lower learning rate.
    """
    tokens: BatchEncoding = TOKENIZER(
        texts, padding=True, truncation=True, return_tensors="pt")

    # actual tensor
    sigmoid_tensor = generate_embeddings_with_gradient(tokens)

    # shape at this moment: [batch_size, seq_length, hidden_size]
    # last_hidden_state = embeddings.last_hidden_state

    # Average over seq_length and hidden_size, shape remaining: (batch_size,)
    # last_hidden_state_mean = last_hidden_state.mean(dim=[1, 2])

    # calculate target tensor
    # Scale user_feedback to a range [0, 1]
    # ts = sigmoid_tensor * user_feedback
    # scaled_target_tensor = (ts - 0.5) * 0.5 + 0.5

    target_tensor = torch.tensor(user_feedback, dtype=torch.float32).view(1, 1)

    print(f"ACTUAL : {sigmoid_tensor}")
    print(f"TARGET : {target_tensor}")

    # Compute loss
    """
    Define loss function.
    Use Mean Squared Error to minimize difference between prediction and target.
    """
    loss_fn = nn.MSELoss()
    loss: MSELoss | Tensor = loss_fn(sigmoid_tensor, target_tensor)
    print("Loss before optimization:", loss.item())

    # Backpropagation
    loss.backward()  # Compute gradients
    OPTIMIZER.step()  # Update model parameters (weights) after backpropagation


def embeddings_to_scalar_sum(sigmoid_tensor: FloatTensor) -> Number:
    """
    Convert embeddings to a single scalar number.
    The procedure is:
      1. Perform mean pooling over the sequence dimension using the attention mask.
      2. Compute the mean for each batch item.
      3. Sum these means to produce a single scalar.

    Args:
        tensor_with_attention_mask (Tuple[BaseModelOutput, Tensor]): A tuple containing:
            - embeddings (BaseModelOutput): Model outputs with `last_hidden_state` (shape: [batch_size, seq_length, hidden_size]).
            - attention_mask (Tensor): Attention mask (shape: [batch_size, seq_length]).

    Returns:
        Number: A single scalar computed as described.
    """
    # last_layer: FloatTensor = embeddings.last_hidden_state

    # Expand the attention mask to match embeddings' dimensions
    # [batch_size, seq_length, hidden_size]
    # input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())

    # Apply the attention mask and compute the sum over the sequence dimension
    # [batch_size, hidden_size]
    # sum_embeddings = torch.sum(embeddings * input_mask_expanded, dim=1)
    # [batch_size, 1]
    # sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)

    # Mean pooling over the sequence dimension
    # mean_pooled = sum_embeddings / sum_mask  # [batch_size, hidden_size]

    # Compute the mean for each batch element, then sum the results
    # scalar_sum = mean_pooled.mean(dim=1).sum().item()  # Scalar

    return sigmoid_tensor.item()


def generate_embeddings(tokens: BatchEncoding) -> FloatTensor:
    """
    This function map a list of texts to a numerical representation, such as an embedding or a classification output.
    Aka, generate semantic embeddings for a list of texts.

    - no_grad() prevents PyTorch from tracking and storing gradients during computations (useful when you're performing inference)
    """
    with torch.no_grad():
        return MY_MODEL.forward(tokens)


def generate_embeddings_with_gradient(tokens: BatchEncoding) -> FloatTensor:
    """
    Calculate embeddings with gradient tracking.
    Return mean pooling of token embeddings.
    """
    return MY_MODEL.forward(tokens)


def inspect_tensor(tensor: Tensor):
    print("------------------------------")
    if len(tensor.shape) == 3:  # Tensor must have 3 dimensions to have a batch size
        # [batch_size, seq_length, hidden_size]
        print("The tensor is 3D.")
        batch_size, seq_len, hidden_size = tensor.shape
        print(f"Batch size: {batch_size}")
        print(f"Number of tokens (sequence length): {seq_len}")
        print(f"Hidden size: {hidden_size}")
    elif len(tensor.shape) == 2:  # 2D tensor
        # [seq_length, hidden_size]
        print("The tensor is 2D (no batch dimension).")
        seq_len, hidden_size = tensor.shape
        print(f"Number of tokens (sequence length): {seq_len}")
        print(f"Hidden size: {hidden_size}")
    elif len(tensor.shape) == 1:  # 1D tensor
        # [hidden_size]
        print("The tensor is 1D (single token embedding, aka only one token, aka sequence length is 1).")
        hidden_size = tensor.shape[0]
        print(f"Hidden size: {hidden_size}")
    else:
        print("Unexpected tensor shape:", tensor.shape)
    print("------------------------------")
