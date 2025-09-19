# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from model.model_predictor import predict_fake_news_xlm, predict_final_combined
from fact_checker import fact_check
from facebook_scraper import scrape_facebook_fallback
import requests

app = FastAPI(title="Fake news detection API Malagasy - Modèle Combiné")

# CORS permissif (tout autoriser). A durcir si mise en prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FacebookURL(BaseModel):
    facebook_url: str
    
    def __init__(self, **data):
        super().__init__(**data)
        # Validation : il faut facebook_url
        if not self.facebook_url:
            raise ValueError("Il faut fournir le 'facebook_url'")

@app.post("/check")
def check_text(data: FacebookURL):
    """
    Analyse d'une publication Facebook pour détecter les fake news
    avec modèle combiné (NLP + Temporel + Random Forest)
    
     - Analyse d'une publication Facebook:
    {
        "facebook_url": "https://www.facebook.com/page/posts/123456789"
    }
    """

    res = requests.get(data.facebook_url, allow_redirects=True)

    # Scraper la publication Facebook
    post_text, formatted_date = scrape_facebook_fallback(res.url)

    # Extraire le texte et la date
    text = post_text
    extracted_date = formatted_date
    
    if not text:
        raise HTTPException(status_code=400, detail="Aucun texte trouvé dans la publication Facebook")
    
    # Utiliser la date de référence fournie ou celle extraite de la publication
    date_reference = extracted_date
    
    if not date_reference:
        raise HTTPException(
            status_code=400, 
            detail="Aucune date de référence fournie et impossible d'extraire la date de la publication"
        )
    
    # === NOUVELLE APPROCHE : Prédiction combinée ===
    combined_result = predict_final_combined(text, date_reference)
    
    # Récupérer les informations pour la compatibilité avec l'ancien système
    final_prediction = combined_result["final_prediction"]
    final_confidence = combined_result["final_confidence"]
    
    # Déterminer is_real_news_nlp pour le fact_checker
    is_real_news_nlp = 1 if final_prediction == "REAL" else 0
    
    # Vérification des faits (optionnelle, peut être désactivée si non nécessaire)
    try:
        fact_check_result = fact_check(text, date_reference, is_real_news_nlp)
    except Exception as e:
        print(f"Erreur fact check: {e}")
        fact_check_result = {
            "status": "Erreur lors de la vérification",
            "similarity": 0.0,
            "url": None,
            "date": None
        }
    
    result = {
        "source": "facebook",
        "facebook_url": data.facebook_url,
        "extracted_text": text,
        "extracted_date": extracted_date,
        "date_reference_used": date_reference,
        
        # === Résultats du modèle combiné ===
        "prediction": final_prediction,
        "confidence": round(final_confidence, 2),
        "model_details": {
            "final_probability": {
                "FAKE": round(combined_result["final_probability"]["FAKE"] * 100, 2),
                "REAL": round(combined_result["final_probability"]["REAL"] * 100, 2)
            },
            "scores": {
                "nlp_score": round(combined_result["scores"]["nlp_score"], 4),
                "temporal_score": round(combined_result["scores"]["temporal_score"], 4)
            },
            "nlp_only": {
                "prediction": combined_result["nlp_details"]["label"],
                "confidence": round(combined_result["nlp_details"]["confidence"], 2)
            }
        },
        
        # === Fact checking (amélioré) ===
        "fact_check": {
            "status": fact_check_result["status"],
            "similarity_score": fact_check_result["similarity"],
            "source_url": fact_check_result["url"] or "Aucune source fiable trouvée",
            "article_date": fact_check_result["date"],
            "source_type": fact_check_result.get("source_type", "unknown"),
            "debug_info": fact_check_result.get("debug_info", {})
        }
    }
    return result

@app.post("/check_legacy")
def check_text_legacy(data: FacebookURL):
    """
    Version legacy utilisant uniquement le modèle NLP XLM-RoBERTa
    """
    # Scraper la publication Facebook
    post_text, formatted_date = scrape_facebook_fallback(data.facebook_url)

    # Extraire le texte et la date
    text = post_text
    extracted_date = formatted_date
    
    if not text:
        raise HTTPException(status_code=400, detail="Aucun texte trouvé dans la publication Facebook")
    
    date_reference = extracted_date
    
    if not date_reference:
        raise HTTPException(
            status_code=400, 
            detail="Aucune date de référence fournie et impossible d'extraire la date de la publication"
        )
    
    # Analyser le texte avec le modèle NLP uniquement
    label, confidence = predict_fake_news_xlm(text)
    is_real_news_nlp = 0 if label == "FAKE" else 1
    
    # Vérification des faits
    fact_check_result = fact_check(text, date_reference, is_real_news_nlp)
    
    result = {
        "source": "facebook",
        "facebook_url": data.facebook_url,
        "extracted_text": text,
        "extracted_date": extracted_date,
        "date_reference_used": date_reference,
        "prediction": label,
        "confidence": round(confidence, 2),
        "fact_check": {
            "status": fact_check_result["status"],
            "similarity_score": fact_check_result["similarity"],
            "source_url": fact_check_result["url"] or "Aucune source fiable trouvée",
            "article_date": fact_check_result["date"],
            "source_type": fact_check_result.get("source_type", "unknown"),
            "debug_info": fact_check_result.get("debug_info", {})
        }
    }
    return result

@app.get("/")
def read_root():
    return {
        "message": "Fake news detection API Malagasy avec modèle combiné (NLP + Temporel + Random Forest)",
        "endpoints": {
            "/check": "Analyser avec le modèle combiné (recommandé)",
            "/check_legacy": "Analyser avec le modèle NLP seul (version legacy)",
            "/health": "Vérifier l'état de l'API"
        },
        "modeles": {
            "nlp": "XLM-RoBERTa fine-tuné",
            "temporel": "Prophet (analyse temporelle)",
            "meta_modele": "Random Forest (combinaison des scores)"
        },
        "exemples": {
            "publication_facebook": {
                "facebook_url": "https://www.facebook.com/page/posts/123456789"
            }
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API opérationnelle avec modèle combiné"
    }
