import pandas as pd
import numpy as np
import torch
import shap
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import matplotlib.pyplot as plt
# Allocate gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model and tokenizer from the specified checkpoint
checkpoint_dir = "path-to-checkpoint"
model = DistilBertForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(checkpoint_dir)
model.eval()

# Model prediction function
def model_predict(input_ids):
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = torch.ones_like(input_ids).to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities.cpu().numpy()

# Load the test dataset
file_path = "/path-to-test_dataset"
test_df = pd.read_excel(file_path, engine='openpyxl')

# Choose balanced random samples
suicide_df = test_df[test_df['class'] == 'suicide'].sample(20)
non_suicide_df = test_df[test_df['class'] == 'non-suicide'].sample(20)
balanced_sample_df = pd.concat([suicide_df, non_suicide_df])

# Tokenize the samples
sampled_texts = balanced_sample_df['text'].tolist()
encodings = tokenizer(sampled_texts, return_tensors="pt", padding=True, truncation=True).to(device)

# SHAP Explainer and calculate the values
explainer = shap.KernelExplainer(model_predict, encodings['input_ids'].cpu().numpy())
shap_values = explainer.shap_values(encodings['input_ids'].cpu().numpy(), nsamples=100)

# Visualisation
shap.summary_plot(shap_values, feature_names=tokenizer.convert_ids_to_tokens(encodings['input_ids'][0].cpu().numpy()))
plt.savefig("path-to-save-shap.png")
#%%
