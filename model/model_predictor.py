# model_predictor.py
import os
import torch
import pandas as pd
import numpy as np
import joblib
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from prophet.serialize import model_from_json

# === Chemins des modèles ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "xlmroberta_finetuned")
PROPHET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "prophet_model.json")
RF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_meta_model.joblib")

# === Vérifications d'existence des fichiers ===
if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"Le répertoire du modèle XLM-RoBERTa '{MODEL_DIR}' n'existe pas")

if not os.path.exists(PROPHET_MODEL_PATH):
    raise FileNotFoundError(f"Le modèle Prophet '{PROPHET_MODEL_PATH}' n'existe pas")

if not os.path.exists(RF_MODEL_PATH):
    raise FileNotFoundError(f"Le modèle Random Forest '{RF_MODEL_PATH}' n'existe pas")

# === Chargement des modèles ===
# 1. Modèle XLM-RoBERTa
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)
nlp_model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_DIR)
nlp_model.eval()

# 2. Modèle Prophet
with open(PROPHET_MODEL_PATH, 'r') as f:
    prophet_model = model_from_json(f.read())

# 3. Modèle Random Forest (méta-modèle)
rf_model = joblib.load(RF_MODEL_PATH)

# === Préparation des données temporelles ===
# Générer les prédictions Prophet pour l'interpolation
future = prophet_model.make_future_dataframe(periods=0)
forecast = prophet_model.predict(future)
forecast['ds'] = pd.to_datetime(forecast['ds']).dt.normalize()
forecast.set_index('ds', inplace=True)

# Normaliser yhat en [0,1] pour le score temporel
yhat = forecast['yhat'].values
yhat_min, yhat_max = yhat.min(), yhat.max()
forecast['score_temporel'] = (yhat - yhat_min) / (yhat_max - yhat_min) if yhat_max != yhat_min else 0.5

# === Fonctions utilitaires ===

def predict_fake_news_xlm(text):
    """
    Prédit si un texte est une fake news ou non avec le modèle XLM-RoBERTa.
    
    Args:
        text (str): Le texte à analyser
        
    Returns:
        tuple: (label, confidence) où label est 'FAKE' ou 'REAL'
               et confidence est le pourcentage de confiance
    """
    try:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=300, padding=True)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
        
        label = 'FAKE' if pred == 0 else 'REAL'
        confidence = float(probs[pred] * 100)
        return label, confidence
    
    except Exception as e:
        print(f"Erreur lors de la prédiction NLP: {e}")
        return "Erreur", 0.0

def predict_nlp_score(text):
    """
    Calcule le score NLP normalisé [0,1] pour le méta-modèle.
    0 = très probablement FAKE, 1 = très probablement REAL
    
    Args:
        text (str): Le texte à analyser
        
    Returns:
        float: Score entre 0 et 1
    """
    try:
        label, confidence = predict_fake_news_xlm(text)
        confidence_normalized = confidence / 100.0
        
        if label == 'FAKE':
            return confidence_normalized  # Plus la confiance est haute pour FAKE, plus le score est bas
        elif label == 'REAL':
            return 1 - confidence_normalized  # Plus la confiance est haute pour REAL, plus le score est haut
        else:
            return 0.5  # En cas d'erreur
            
    except Exception as e:
        print(f"Erreur lors du calcul du score NLP: {e}")
        return 0.5

def get_temporal_score(date):
    """
    Calcule le score temporel basé sur le modèle Prophet.
    
    Args:
        date (str ou datetime): Date à analyser
        
    Returns:
        float: Score temporel entre 0 et 1
    """
    try:
        dt = pd.to_datetime(date).normalize()
        
        # Si la date existe directement dans le forecast
        if dt in forecast.index:
            return float(forecast.loc[dt, 'score_temporel'])
        
        # Interpolation linéaire simple
        before = forecast.index[forecast.index <= dt].max()
        after = forecast.index[forecast.index >= dt].min()
        
        if pd.isna(before) or pd.isna(after):
            return 0.5  # Score neutre si pas de données
        
        if before == after:
            return float(forecast.loc[before, 'score_temporel'])
        
        s_before = float(forecast.loc[before, 'score_temporel'])
        s_after = float(forecast.loc[after, 'score_temporel'])
        total = (after - before).days
        frac = (dt - before).days / total if total > 0 else 0
        
        return s_before + frac * (s_after - s_before)
        
    except Exception as e:
        print(f"Erreur lors du calcul du score temporel: {e}")
        return 0.5

def predict_final_combined(text, date):
    """
    Prédiction finale combinant les modèles NLP, temporel et Random Forest.
    
    Args:
        text (str): Le texte à analyser
        date (str ou datetime): Date de référence
        
    Returns:
        dict: Résultats de la prédiction avec tous les scores
    """
    try:
        # 1. Calculer les scores individuels
        score_nlp = predict_nlp_score(text)
        score_temporel = get_temporal_score(date)
        
        # 2. Prédiction avec le méta-modèle Random Forest
        X = np.array([[score_nlp, score_temporel]])
        pred = rf_model.predict(X)[0]
        proba = rf_model.predict_proba(X)[0]
        
        # 3. Récupérer aussi les résultats détaillés du modèle NLP
        nlp_label, nlp_confidence = predict_fake_news_xlm(text)
        
        # 4. Construire le résultat
        result = {
            "final_prediction": pred,
            "final_probability": {
                "FAKE": float(proba[0]),
                "REAL": float(proba[1])
            },
            "final_confidence": float(max(proba) * 100),
            "scores": {
                "nlp_score": float(score_nlp),
                "temporal_score": float(score_temporel)
            },
            "nlp_details": {
                "label": nlp_label,
                "confidence": float(nlp_confidence)
            }
        }
        
        return result
        
    except Exception as e:
        print(f"Erreur lors de la prédiction finale: {e}")
        return {
            "final_prediction": "Erreur",
            "final_probability": {"FAKE": 0.5, "REAL": 0.5},
            "final_confidence": 0.0,
            "scores": {"nlp_score": 0.5, "temporal_score": 0.5},
            "nlp_details": {"label": "Erreur", "confidence": 0.0},
            "error": str(e)
        }
