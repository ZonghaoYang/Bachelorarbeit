import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from google.colab import drive
drive.mount('/content/drive')

# Load the test set data
df = pd.read_csv('/content/drive/MyDrive/TestSet_Combined_unshuffled.tsv', delimiter='\t')

# Initialize the model and tokenizer
model_name = "zonghaoyang/DistilRoBERTa-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Get prediction score and exclude texts that are too long:
def get_prediction_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    if len(inputs["input_ids"][0]) > 512:
        return None
    with torch.no_grad():
        logits = model(**inputs).logits
    # Get the score for the 'Association' class
    association_score = torch.softmax(logits, dim=-1)[0][1].item()
    return association_score

# Apply the function to the 'text' column
df['score'] = df['text'].apply(get_prediction_score)
# Drop the rows where 'score' is None
df = df.dropna(subset=['score'])

# Find the maximum score for each gene pair
max_scores = df.groupby(['gene1', 'gene2'])['score'].max()
# Apply the maximum score to each gene pair
df['max_score'] = df.apply(lambda row: max_scores[(row['gene1'], row['gene2'])], axis=1)
# Predict the label based on the maximum score
df['predicted_label'] = np.where(df['max_score'] > 0.5, 'Association', 'None')

from sklearn.metrics import f1_score

y_true = df['label']
y_pred = df['predicted_label']

f1 = f1_score(y_true, y_pred, pos_label='Association', average='binary')

print('F1 Score: ', f1)