# model_predictor.py
import os
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Obtenir le chemin absolu du répertoire du modèle
MODEL_DIR = os.path.join(os.path.dirname(__file__), "xlmroberta_finetuned")

# Vérifier si le répertoire du modèle existe
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Le répertoire du modèle '{MODEL_DIR}' n'existe pas")

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

def predict_fake_news_xlm(text):
    """
    Prédit si un texte est une fake news ou non.
    
    Args:
        text (str): Le texte à analyser
        
    Returns:
        tuple: (label, confidence) où label est 'Fake News' ou 'Real News'
               et confidence est le pourcentage de confiance
    """
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=300, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
        
        label = 'FAKE' if pred == 0 else 'REAL'
        confidence = float(probs[pred] * 100)
        return label, confidence
    
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return "Erreur", 0.0
