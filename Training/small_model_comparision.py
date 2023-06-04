from huggingface_hub import notebook_login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from datasets import load_metric
import numpy as np

# Login to the hugging face
notebook_login()

BioRED = load_dataset("zonghaoyang/BioRED", "BioRED")

# Filter the text instances that contain 2 occurrences of '@GeneOrGeneProduct$' for gene pairs
def filter_gene_gene(examples):
    return {'keep': [text.count('@GeneOrGeneProduct$') == 2 for text in examples['text']]}

# Extract datasets
biored_train = BioRED['train'].filter(filter_gene_gene)
biored_valid = BioRED['validation'].filter(filter_gene_gene)
biored_test = BioRED['test'].filter(filter_gene_gene)

# Load a DistilBERT tokenizer to preprocess the text
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Create a map of the expected ids to their labels with id2label and label2id
id2label = {0: "None", 1: "Association"}
label2id = {"None": 0, "Association": 1}

def preprocess_function(examples):
    labels = ['Association' if label == 'Association' else 'None' for label in examples["label"]]
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    encoded_labels = [label2id[label] for label in labels]
    examples["label"] = encoded_labels
    return {**tokenized_inputs, **examples}

tokenized_biored_train = biored_train.map(preprocess_function, batched=True)
tokenized_biored_valid = biored_valid.map(preprocess_function, batched=True)
tokenized_biored_test = biored_test.map(preprocess_function, batched=True)

# Create a batch of examples using DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Evaluation metrics
accuracy = load_metric("accuracy")
f1_score = load_metric("f1")
precision = load_metric("precision")
recall = load_metric("recall")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        "accuracy": accuracy.compute(predictions=predictions, references=labels)['accuracy'],
        "f1": f1_score.compute(predictions=predictions, references=labels)['f1'], 
        "precision": precision.compute(predictions=predictions, references=labels)['precision'],
        "recall": recall.compute(predictions=predictions, references=labels)['recall'],
    }

# Create smaller training set
train_split_ratio = 0.5
biored_train_small = biored_train.train_test_split(train_size=train_split_ratio)['train']
tokenized_biored_train_small = biored_train_small.map(preprocess_function, batched=True)

# Load DistilBERT with AutoModelForSequenceClassification along with the number of expected labels, and the label mappings
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="DistilBERT-smaller-BioRED2e-05",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True,
    push_to_hub=True,
)

# Clone the model for training on smaller dataset
model_small = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

# Training on smaller dataset
trainer_small = Trainer(
    model=model_small,
    args=training_args,
    train_dataset=tokenized_biored_train_small,
    eval_dataset=tokenized_biored_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer_small.train()

eval_results_small = trainer_small.evaluate(tokenized_biored_test)

# Share the model
trainer_small.push_to_hub()