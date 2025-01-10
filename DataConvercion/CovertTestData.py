import pandas as pd
import re

data = pd.read_csv('dataToConversion/test_recipes.csv')

def extract_nutrition_values(nutrition_str):
    fat_pattern = r'Total Fat (\d+)g'
    carbs_pattern = r'Total Carbohydrate (\d+)g'
    protein_pattern = r'Protein (\d+)g'

    fat = int(re.search(fat_pattern, nutrition_str).group(1)) if re.search(fat_pattern, nutrition_str) else 0
    carbs = int(re.search(carbs_pattern, nutrition_str).group(1)) if re.search(carbs_pattern, nutrition_str) else 0
    protein = int(re.search(protein_pattern, nutrition_str).group(1)) if re.search(protein_pattern, nutrition_str) else 0

    return fat, carbs, protein

def calculate_calories(fat, carbs, protein):
    return fat * 9 + carbs * 4 + protein * 4

data['calories'] = data['nutrition'].apply(lambda x: calculate_calories(*extract_nutrition_values(x)))

data.to_csv('data/modified_recipes.csv', index=False)