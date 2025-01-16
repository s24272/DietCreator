from phi.agent import Agent
from phi.model.huggingface import HuggingFaceModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "./fine-tuned-flan-t5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class DietGeneratorAgent(Agent):
    def __init__(self, model, tokenizer):
        super().__init__(
            model=HuggingFaceModel(model=model, tokenizer=tokenizer),
            markdown=True,
        )

    def generate_diet(self, prompt):
        response = self.model.generate(prompt)
        return response

class DietVerifierAgent:
    def verify_diet(self, diet, requirements):
        total_calories = sum(recipe["calories"] for recipe in diet)
        if total_calories != requirements["calories"]:
            return False, "Suma kalorii nie zgadza się z wymaganiami."

        if len(diet) != requirements["meals"]:
            return False, "Liczba posiłków nie zgadza się z wymaganiami."

        for recipe in diet:
            if "meat" in recipe["ingredients"] and not requirements["meat"]:
                return False, "Przepis zawiera mięso, mimo że dieta ma być bezmięsna."

        return True, "Przepisy spełniają wymagania."

if __name__ == "__main__":
    generator_agent = DietGeneratorAgent(model, tokenizer)
    verifier_agent = DietVerifierAgent()

    user_prompt = input("Podaj swoje wymagania dotyczące diety: ")
    requirements = {
        "meat": "mięs" not in user_prompt.lower(),
        "meals": int(input("Podaj liczbę posiłków: ")),
        "calories": int(input("Podaj liczbę kalorii: ")),
    }

    diet = generator_agent.generate_diet(user_prompt)
    is_valid, message = verifier_agent.verify_diet(diet, requirements)

    if not is_valid:
        print(f"Błąd: {message}")
        diet = generator_agent.generate_diet(user_prompt)
    else:
        print("Dieta spełnia wymagania:")
        for recipe in diet:
            print(f"- {recipe['name']} ({recipe['calories']} kcal)")

    user_question = input("Czy chcesz poznać przepis na konkretną potrawę? (Tak/Nie): ")
    if user_question.lower() == "tak":
        recipe_name = input("Podaj nazwę potrawy: ")
        recipe_response = generator_agent.generate_diet(f"Jak przygotować {recipe_name}?")
        print(recipe_response)