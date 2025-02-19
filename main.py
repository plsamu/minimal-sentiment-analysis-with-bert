from mllib import compute_backpropagation, make_prediction

texts = [
    "Semantic embeddings are useful for text representation."
]

print("----------------------\n")
predicted_sentiment = make_prediction(texts)
print(f"First prediction: {predicted_sentiment}")
compute_backpropagation(texts, 1)

# Recalculate embeddings and sentiment value after feedback
print("----------------------\n")
predicted_sentiment = make_prediction(texts)
print(f"Second prediction: {predicted_sentiment}")
compute_backpropagation(texts, 1)

print("----------------------\n")
predicted_sentiment = make_prediction(texts)
print(f"Third prediction: {predicted_sentiment}")
