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

        # Get model config to get last layer
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

# Define optimizer (only optimize last_layer) using a smaller that default learning rate
# AdamW decouples weight decay from the gradient update step (reduces overfitting)
# parameters: includes learnable weights and biases
# parameters() call will always return references to the same tensors so this object can be static
OPTIMIZER = AdamW(MY_MODEL.last_layer.parameters(), lr=1e-5)

def make_prediction(texts: List[str]) -> Number:
    tokens: BatchEncoding = TOKENIZER(
        texts, padding=True, truncation=True, return_tensors="pt")
    sigmoid_tensor = generate_embeddings(tokens)
    return sigmoid_tensor.item()


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

    # create a target tensore using user_feedback and expand to the right shape
    # tensor(1, dtype=torch.float32) -> tensor([1.]) -> view(1, 1) -> tensor([[1.]])
    target_tensor = torch.tensor(user_feedback, dtype=torch.float32).view(1, 1)

    print(f"ACTUAL : {sigmoid_tensor}")
    print(f"TARGET : {target_tensor}")

    """
    Compute and define loss function.
    Use Mean Squared Error to minimize difference between prediction and target.
    """
    loss_fn = nn.MSELoss()
    loss: MSELoss | Tensor = loss_fn(sigmoid_tensor, target_tensor)
    print("Loss before optimization:", loss.item())

    # Backpropagation
    loss.backward()  # Compute gradients
    OPTIMIZER.step()  # Update model parameters (weights) after backpropagation


def generate_embeddings(tokens: BatchEncoding) -> FloatTensor:
    """
    This function map a list of texts to a numerical representation, aka semantic embeddings or a classification output.

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
