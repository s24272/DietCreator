import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

data = pd.read_csv("archive/recipes_cleaned.csv")

dataset = Dataset.from_pandas(data)

dataset = dataset.map(lambda x: {
    "text": (x.get("title", "") if x.get("title") is not None else "") +
            " Ingredients: " + ", ".join(eval(x.get("ingredients", "[]"))) +
            " Instructions: " + x.get("instructions", ""),
    "instructions": x.get("instructions", "")
})

train_test_split = dataset.train_test_split(test_size=0.1, train_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["instructions"], padding="max_length", truncation=True, max_length=512)
    inputs["labels"] = labels["input_ids"]
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    bleu_metric = load("bleu")
    rouge_metric = load("rouge")

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    print("Predictions:", predictions[:5])
    print("Labels:", labels[:5])

    bleu = bleu_metric.compute(predictions=predictions, references=labels)["bleu"]
    rouge = rouge_metric.compute(predictions=predictions, references=labels)["rougeL"]

    y_test = [1 if label else 0 for label in labels]
    y_pred = [1 if pred else 0 for pred in predictions]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "bleu": bleu,
        "rouge": rouge,
    }

    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values())
    plt.title("Model Metrics")
    plt.ylabel("Score")
    plt.show()

    return metrics

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model("./fine-tuned-flan-t5")
tokenizer.save_pretrained("./fine-tuned-flan-t5")

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")