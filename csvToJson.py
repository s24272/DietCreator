import pandas as pd
import json
import nltk


nltk.download('punkt')

df = pd.read_csv("archive/recipes_cleaned.csv")
df.to_json("archive/modified_recipes.json", orient="records", lines=True)