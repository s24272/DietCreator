import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

df = pd.read_csv("DataConvercion/data/modified_recipes2.csv")

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

df['text'] = df['recipe_name'] + " Ingredients: " + df['ingredients'].apply(
    lambda x: ", ".join([f"{ing['quantity']} {ing['unit']} {ing['name']}" for ing in parse_ingredients(x)])
)

dataset = Dataset.from_pandas(df[['text']])

model_name = "meta-llama/Llama-3.1-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

model = AutoModelForCausalLM.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()

trainer.save_model("./fine-tuned-llama")
tokenizer.save_pretrained("./fine-tuned-llama")