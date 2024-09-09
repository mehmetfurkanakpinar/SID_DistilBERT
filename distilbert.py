#%%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import  AutoModelForSequenceClassification, AutoTokenizer
checkpoint = "/data1/ma2/checkpoints/checkpoint-146178"
classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("Model and tokenizer is loaded")
classifier = classifier.to(device)
#%%
import pandas as pd
file_path = "/data1/ma2/train_set_185k.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
#%%
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])
print("Labels are encoded")
#%%
from sklearn.model_selection import train_test_split
df_train, df_eval = train_test_split(df, train_size=0.7,stratify=df["class"], random_state=42)
print("Dataset split into train and eval")
#%%
from datasets import Dataset, DatasetDict
raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(df_train),
    "eval": Dataset.from_pandas(df_eval)
})
#%%
print("Dataset Dict:\n", raw_datasets)
print("\n\nTrain's features:\n", raw_datasets["train"].features)
print("\n\nFirst row of Train:\n", raw_datasets["train"][0])
#%%
raw_datasets = raw_datasets.map(lambda dataset: {'text': str(dataset['text'])}, batched=False)
tokenized_datasets = raw_datasets.map(lambda dataset: tokenizer(dataset['text'], truncation=True), batched=True)
#%%
print(tokenized_datasets)
#%%
print(tokenized_datasets["train"][0])
#%%
tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("class", "labels")
print(tokenized_datasets)
#%%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#%%
checkpoint_dir = os.path.join("/data1/ma2", "checkpoints")

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    print(f"Checkpoint file is created: {checkpoint_dir}")
else:
    print(f"Checkpoint file already exists: {checkpoint_dir}")
#%%
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import evaluate

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training args
training_args = TrainingArguments(
    output_dir=checkpoint_dir,              # Checkpoint'lerin kaydedileceği dizin
    num_train_epochs=10,                    # Eğitim epoch sayısı
    eval_strategy="epoch",            # Her epoch sonunda değerlendirme yapılacak
    weight_decay=1e-3,                      # Ağırlık çürüme oranı
    save_strategy="epoch",                  # Her epoch sonunda model kaydedilecek
    save_total_limit=5,                     # En fazla 3 checkpoint saklanacak
    report_to="none",                       # Eğitim raporlama (örneğin WandB) kapalı
    load_best_model_at_end=True,            # Eğitim sonunda en iyi modeli yükle
    metric_for_best_model="accuracy",
    logging_dir = './logs',
    logging_steps=500
)

# Metric for validation error
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc") # F1 and Accuracy
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainerCallback

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            self.train_loss.append(logs['loss'])
        if 'eval_loss' in logs:
            self.val_loss.append(logs['eval_loss'])
            self.epochs.append(state.epoch)

loss_history = LossHistoryCallback()
# Define trainer
trainer = Trainer(
    classifier,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["eval"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2),loss_history] # Early Stopping Callback'i ekle
)
#%%
trainer.train(resume_from_checkpoint=checkpoint)

