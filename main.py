import json
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    RobertaTokenizer,
    RobertaModel,  # <--- CHANGED: We need the base model
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
# <--- NEW IMPORTS START --->
import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
# <--- NEW IMPORTS END --->
from torch.utils.data import Dataset
import os

# ------------------ Load Dataset ------------------
try:
    with open("/content/drive/MyDrive/Sarcasm_Headlines_Dataset.json", 'r') as f:
        data = [json.loads(line) for line in f]
except FileNotFoundError:
    print("Error: Dataset file not found. Please update the file path.")
    # In a real script, you might want to exit or raise an error
    # For this example, we'll create dummy data if file not found
    data = [
        {"headline": "Man truly loved by his dog", "is_sarcastic": 0},
        {"headline": "Area man constantly pointing out he's being sarcastic", "is_sarcastic": 1},
        {"headline": "This is a great day", "is_sarcastic": 0},
        {"headline": "Oh, wonderful, another meeting", "is_sarcastic": 1}
    ] * 1000
    print("Warning: Dataset not found. Using dummy data.")

df = pd.DataFrame(data)
texts = df['headline']
labels = df['is_sarcastic']
print("hi")

# ------------------ Tokenization ------------------
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

X_train, X_test, y_train, y_test = train_test_split(
    df['headline'].values,
    labels.values,
    test_size=0.2,
    random_state=42
)

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = SarcasmDataset(X_train, y_train, tokenizer)
test_dataset = SarcasmDataset(X_test, y_test, tokenizer)



class RobertaBiLSTMMHA(nn.Module):
    """
    Custom model combining RoBERTa, Bi-LSTM, and Multi-Head Attention.
    This class is modified to be compatible with the transformers.Trainer.
    """
    def __init__(self, num_classes, roberta_model_name='roberta-large', hidden_dim=256, n_layers=2, n_heads=8, dropout=0.1):
        super(RobertaBiLSTMMHA, self).__init__()

        self.num_classes = num_classes

        # 1. RoBERTa Layer
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        roberta_hidden_size = self.roberta.config.hidden_size

        # 2. Bi-LSTM Layer
        self.lstm = nn.LSTM(
            input_size=roberta_hidden_size,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # 3. Multi-Head Attention Layer
        lstm_output_dim = hidden_dim * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # 4. Classification Head
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

        # Utilities
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass modified for transformers.Trainer compatibility.

        It now accepts 'labels' and returns loss + logits.
        """

        # 1. RoBERTa
        roberta_output = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = roberta_output.last_hidden_state
        # shape: (batch_size, seq_len, 768)

        # 2. Bi-LSTM
        lstm_output, (hidden, cell) = self.lstm(sequence_output)
        # shape: (batch_size, seq_len, hidden_dim * 2)

        # 3. Multi-Head Attention
        padding_mask = (attention_mask == 0) # MHA needs True for padded tokens
        attn_output, _ = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=padding_mask
        )
        # shape: (batch_size, seq_len, hidden_dim * 2)

        # 4. Pooling & Classification
        # We use the [CLS] token output (first token)
        cls_token_output = attn_output[:, 0, :]
        cls_token_output = self.layer_norm(cls_token_output)
        cls_token_output = self.dropout(cls_token_output)

        # 5. Classifier
        logits = self.classifier(cls_token_output)
        # shape: (batch_size, num_classes)

        # 6. Calculate Loss (if labels are provided)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        # Return Trainer-compatible output
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=roberta_output.hidden_states, # Optional
            attentions=roberta_output.attentions,       # Optional
        )

# Instantiate the new custom model
model = RobertaBiLSTMMHA(num_classes=2, roberta_model_name='roberta-large')

# <--- NEW CUSTOM MODEL CLASS END --->

# ------------------ Metrics Function ------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
    }

# ------------------ Training ------------------
training_args = TrainingArguments(
    output_dir='./results-custom-model', # Changed output dir
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir='./logs-custom-model', # Changed log dir
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

# ------------------ Evaluation ------------------
print("Evaluating the best model...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)

print("Classification Report:")
print(classification_report(y_test, y_pred))
