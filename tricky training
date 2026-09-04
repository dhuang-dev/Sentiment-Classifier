# ============================================================
# Fine-tune further on hand-labeled "hard case" examples
# (double negatives, implied sentiment, etc.)
# Run in Google Colab (needs internet access to Hugging Face)
# ============================================================

# !pip install transformers datasets evaluate accelerate -q

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# --- 1. Load your hand-labeled CSV ---
# Upload tricky_examples.csv to Colab first (folder icon on the left -> upload)
df = pd.read_csv("tricky_examples.csv")
print(df.head())
print(f"\nTotal examples: {len(df)}")

# With only ~25 examples, don't hold out a separate test set —
# there's too little data to split meaningfully. We'll train on
# all of it and sanity-check with hand-picked sentences at the end.
dataset = Dataset.from_pandas(df)

# --- 2. Start from the ALREADY sentiment-trained model, not raw DistilBERT ---
# This matters: raw distilbert-base-uncased knows nothing about sentiment yet.
# Starting from the sst2 checkpoint means we're refining existing sentiment
# knowledge with your hard examples, not starting from zero.
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# --- 3. Tokenize ---
def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

dataset = dataset.map(tokenize_fn, batched=True)

# --- 4. Training configuration ---
# Key differences from the IMDB script:
# - Much lower learning rate (1e-5 instead of 2e-5): the model already
#   knows general sentiment, we want to nudge it gently on hard cases,
#   not overwrite what it already learned correctly.
# - More epochs (10): with so few examples, one pass barely moves the
#   weights. We need to see these examples several times for gradient
#   descent to meaningfully adjust the model on this specific pattern.
training_args = TrainingArguments(
    output_dir="./results-tricky",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_steps=5,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# --- 5. Train (this IS the gradient descent step) ---
# Each epoch: forward pass -> compute loss -> backward pass computes
# gradients -> optimizer (AdamW) nudges every weight slightly to reduce
# loss on these examples. Ten epochs = ten rounds of this nudging.
trainer.train()

# --- 6. Save the improved model ---
model.save_pretrained("./my-improved-sentiment-model")
tokenizer.save_pretrained("./my-improved-sentiment-model")

# --- 7. Sanity check on new sentences (not in the training data) ---
from transformers import pipeline
classifier = pipeline("text-classification", model="./my-improved-sentiment-model", tokenizer=tokenizer)

test_sentences = [
    "This isn't half bad.",
    "I need to buy five more of these for my whole family.",
    "Not the disaster I was expecting.",
    "This is the third time I've had to return one of these.",
]

for sentence in test_sentences:
    result = classifier(sentence)[0]
    print(f"{sentence!r} -> {result['label']} ({result['score']:.2%})")
