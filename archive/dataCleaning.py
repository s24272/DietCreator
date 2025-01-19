import pandas as pd

data = pd.read_csv("archive/recipes.csv")

columns_to_drop = [
    "AuthorId", "AuthorName", "CookTime", "PrepTime",
    "DatePublished", "Images", "AggregatedRating",
    "ReviewCount", "RecipeServings", "RecipeYield"
]

data = data.drop(columns=columns_to_drop)

data.to_csv("archive/recipes_cleaned.csv", index=False)

