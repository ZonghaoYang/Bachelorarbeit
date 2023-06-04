from huggingface_hub import notebook_login
# Login to Hugging Face
notebook_login()

from datasets import load_dataset
BioRED = load_dataset("zonghaoyang/BioRED", "BioRED")
# Extract train validation and test datasets
biored_train = BioRED['train']
biored_valid = BioRED['validation']
biored_test = BioRED['test']

# Filter the text instances that contain 2 occurrences of '@GeneOrGeneProduct$' for gene pairs
def filter_gene_gene(examples):
    return {'keep': [text.count('@GeneOrGeneProduct$') == 2 for text in examples['text']]}

biored_train = BioRED['train'].filter(filter_gene_gene)
biored_valid = BioRED['validation'].filter(filter_gene_gene)
biored_test = biored_test.filter(filter_gene_gene)


from transformers import AutoTokenizer
# Load BioLinkBERT tokenizer to preprocess the text
tokenizer = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")

def preprocess_function(examples):
    # Change label to 'Association' or 'None'
    labels = ['Association' if label == 'Association' else 'None' for label in examples["label"]]
    
    # Tokenize the text
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    
    # Encode the labels
    encoded_labels = [label2id[label] for label in labels]

    # Update the examples dictionary with the new labels
    examples["label"] = encoded_labels

    # Return tokenized inputs and updated examples
    return {**tokenized_inputs, **examples}

# Create a map of the expected ids to their labels with id2label and label2id
id2label = {0: "None", 1: "Association"}
label2id = {"None": 0, "Association": 1}

tokenized_biored_train = biored_train.map(preprocess_function, batched=True)
tokenized_biored_valid = biored_valid.map(preprocess_function, batched=True)
tokenized_biored_test = biored_test.map(preprocess_function, batched=True)

from transformers import DataCollatorWithPadding
# Create a batch of examples using DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from datasets import load_metric

#Evaluation metrics
accuracy = load_metric("accuracy")
f1_score = load_metric("f1")
precision = load_metric("precision")
recall = load_metric("recall")

import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        "f1": f1_score.compute(predictions=predictions, references=labels)['f1'], 
        "precision": precision.compute(predictions=predictions, references=labels)['precision'],
        "recall": recall.compute(predictions=predictions, references=labels)['recall'],
    }

train_dataset = tokenized_biored_train
eval_dataset = tokenized_biored_valid

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load BioLinkBERT with AutoModelForSequenceClassification along with the number of expected labels, and the label mappings
model = AutoModelForSequenceClassification.from_pretrained(
    "michiyasunaga/BioLinkBERT-base", num_labels=2, id2label=id2label, label2id=label2id
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="BioLinkBERT-base",
    learning_rate=2e-05,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_biored_train,
    eval_dataset=tokenized_biored_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

tokenized_biored_train = tokenized_biored_train.remove_columns(['text', 'label'])
tokenized_biored_valid = tokenized_biored_valid.remove_columns(['text', 'label'])
tokenized_biored_test = tokenized_biored_test.remove_columns(['text', 'label'])

import torch
trainer.train()

test_results = trainer.evaluate(tokenized_biored_test)

# Share the model
trainer.push_to_hub()