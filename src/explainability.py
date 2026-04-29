import os
import torch
import torch.nn.functional as F
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

def run_explainability():

    model_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "saved_models")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "explainability")

    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    base_samples = [
        "I absolutely loved this product, it worked perfectly and arrived fast!",
        "Terrible experience. The item broke after two days of use.",
        "It's okay, not the best but it gets the job done for the price.",
        "Fantastic quality! I will definitely be buying from this seller again.",
        "Worst purchase ever. The customer service was incredibly rude."
    ]
    samples = (base_samples * 4)[:20]

    print(f"Running explanations on {len(samples)} samples...")

    print("\n--- Starting SHAP Analysis ---")
    pred_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    explainer_shap = shap.Explainer(pred_pipe)
    shap_values = explainer_shap(samples)
    
    shap_html_path = os.path.join(output_dir, "shap_explanations.html")
    shap_html = shap.plots.text(shap_values, display=False)
    with open(shap_html_path, "w", encoding="utf-8") as f:
        f.write(shap_html)
    print(f"SHAP explanations saved to: {shap_html_path}")

    print("\n--- Starting LIME Analysis ---")
    lime_explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
    
    def lime_predictor(texts):

        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
            
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        return probs.numpy()

    for i, text in enumerate(samples):
        print(f"Generating LIME explanation {i+1}/{len(samples)}...")
        exp = lime_explainer.explain_instance(text, lime_predictor, num_features=6)
        lime_html_path = os.path.join(output_dir, f"lime_explanation_{i+1}.html")
        exp.save_to_file(lime_html_path)

    print("\nAll explainability tasks complete! Folder recreated successfully.")

if __name__ == "__main__":
    run_explainability()