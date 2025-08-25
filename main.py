# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from model.model_predictor import predict_fake_news_xlm
from fact_checker import fact_check
from facebook_scraper import scrape_facebook_fallback

app = FastAPI(title="Fake news detection API Malagasy")

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
    
     - Analyse d'une publication Facebook:
    {
        "facebook_url": "https://www.facebook.com/page/posts/123456789"
    }
    """
    

    # Scraper la publication Facebook
    post_text, formatted_date = scrape_facebook_fallback(data.facebook_url)

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
    
    # Analyser le texte avec le modèle
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
            "article_date": fact_check_result["date"]
        }
    }
    return result

@app.get("/")
def read_root():
    return {
        "message": "Fake news detection API Malagasy avec support Facebook",
        "endpoints": {
            "/check": "Analyser un texte directement ou une publication Facebook via son URL",
            "/health": "Vérifier l'état de l'API"
        },
        "exemples": {
            "publication_facebook": {
                "facebook_url": "https://www.facebook.com/page/posts/123456789",
                "date_reference": "2025-01-15"
            }
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "API opérationnelle"
    }
