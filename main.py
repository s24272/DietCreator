import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from evaluate import load
import json
import nltk

nltk.download('punkt')

df = pd.read_csv("DataConvercion/data/modified_recipes2.csv")
df.to_json("DataConvercion/data/modified_recipes2.json", orient="records", lines=True)

with open("DataConvercion/data/modified_recipes2.json", "r") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

def parse_ingredients(ingredients_str):
    ingredients_list = ingredients_str.split(", ")
    parsed_ingredients = []
    for ing in ingredients_list:
        parts = ing.split(" ", 2)
        if len(parts) == 3:
            parsed_ingredients.append({
                "quantity": parts[0],
                "unit": parts[1],
                "name": parts[2]
            })
    return parsed_ingredients

dataset = dataset.map(lambda x: {
    "text": x["recipe_name"] + " Ingredients: " + ", ".join([
        f"{ing['quantity']} {ing['unit']} {ing['name']}" for ing in parse_ingredients(x["ingredients"])
    ])
})

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    labels = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
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

    return {
        "bleu": bleu,
        "rouge": rouge,
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
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