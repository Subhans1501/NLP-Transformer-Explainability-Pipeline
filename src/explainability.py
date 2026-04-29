import os
import torch
import torch.nn.functional as F
import shap
import lime
from lime.lime_text import LimeTextExplainer
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def run_explainability():

    model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "explainability")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    samples = [
        "I absolutely loved this product, it worked perfectly and arrived fast!",
        "Terrible experience. The item broke after two days of use.",
        "It's okay, not the best but it gets the job done for the price.",
        "Fantastic quality! I will definitely be buying from this seller again.",
        "Worst purchase ever. The customer service was incredibly rude.",
    ]
    samples = (samples * 4)[:20] 

    print(f"Running explanations on {len(samples)} samples...")


    print("\n--- Starting LIME Analysis ---")
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])

    def lime_predictor(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = F.softmax(logits, dim=-1)
        return probs.numpy()

    for i, text in enumerate(samples):
        print(f"Generating LIME explanation {i+1}/20...")
        exp = lime_explainer.explain_instance(text, lime_predictor, num_features=6)

        lime_html_path = os.path.join(output_dir, f"lime_explanation_{i+1}.html")
        exp.save_to_file(lime_html_path)

    print("\nAll explainability tasks complete!")

if __name__ == "__main__":
    run_explainability()