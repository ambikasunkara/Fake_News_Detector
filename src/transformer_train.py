import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ===============================
# 1. Load Dataset
# ===============================
fake_df = pd.read_csv("data/Fake.csv")
real_df = pd.read_csv("data/True.csv")

fake_df["label"] = 0
real_df["label"] = 1

df = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)
df = df[["text", "label"]]  # keep only needed columns

# ===============================
# 2. Convert to Hugging Face Dataset
# ===============================
dataset = Dataset.from_pandas(df)

# Train/Test Split
dataset = dataset.train_test_split(test_size=0.2)

# ===============================
# 3. Tokenizer
# ===============================
MODEL_NAME = "bert-base-uncased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ===============================
# 4. Model
# ===============================
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# ===============================
# 5. Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2
)

# ===============================
# 6. Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
)

# ===============================
# 7. Train
# ===============================
trainer.train()

# ===============================
# 8. Save Model
# ===============================
model.save_pretrained("model/bert_fake_news")
tokenizer.save_pretrained("model/bert_fake_news")

print("✅ Transformer model trained and saved in model/bert_fake_news/")
