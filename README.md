# 🚀 Fake News Detection API - Backend

## 📋 Vue d'ensemble

API backend pour la détection de fake news en malgache utilisant une approche multi-modèles combinant l'analyse NLP, temporelle et un méta-modèle d'ensemble.

Le système analyse les publications Facebook pour déterminer leur véracité en utilisant trois modèles complémentaires :

- **XLM-RoBERTa** : Analyse sémantique du contenu
- **Prophet** : Analyse des tendances temporelles
- **Random Forest** : Méta-modèle combinant les scores précédents

## 🏗️ Architecture du Système

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│   NLP Model      │───▶│  NLP Score      │
│   (Facebook)    │    │  (XLM-RoBERTa)   │    │   [0,1]         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    │
│   Date Input    │───▶│ Temporal Model   │───▶│ Temporal Score  │    │
│                 │    │   (Prophet)      │    │    [0,1]        │    │
└─────────────────┘    └──────────────────┘    └─────────────────┘    │
                                                         │              │
                                                         ▼              ▼
                                                ┌─────────────────────────┐
                                                │   Meta-Model            │
                                                │  (Random Forest)        │
                                                │                         │
                                                │  Final Prediction:      │
                                                │  FAKE | REAL           │
                                                └─────────────────────────┘
```

## 🛠️ Technologies Utilisées

### Backend

- **FastAPI** : Framework web moderne et rapide
- **Python 3.8+** : Langage principal
- **PyTorch** : Deep learning framework
- **Transformers** : Modèles de langage pré-entraînés
- **Prophet** : Modélisation de séries temporelles
- **Scikit-learn** : Machine learning traditionnel

### Modèles IA

- **XLM-RoBERTa Fine-tuné** : Modèle multilingue pour l'analyse de texte
- **Prophet** : Modèle de prédiction temporelle de Facebook
- **Random Forest** : Méta-modèle d'ensemble

### Outils

- **BeautifulSoup4** : Web scraping
- **Selenium** : Automatisation navigateur
- **Requests** : Requêtes HTTP

## 📁 Structure du Projet

```
API/
├── main.py                     # Point d'entrée FastAPI
├── requirements.txt            # Dépendances Python
├── README.md                  # Documentation
│
├── model/                     # Modèles IA
│   ├── model_predictor.py     # Logique de prédiction combinée
│   ├── xlmroberta_finetuned/  # Modèle NLP fine-tuné
│   ├── prophet_model.json     # Modèle temporel Prophet
│   └── random_forest_meta_model.joblib  # Méta-modèle
│
├── facebook_scraper.py        # Extraction données Facebook
├── fact_checker.py           # Vérification des faits
└── get_article_date.py       # Extraction de dates
```

## 🔧 Installation et Configuration

### 1. Cloner le repository

```bash
git clone https://github.com/Maheni-Ntsoa/fake-news-detection-backend.git
cd fake-news-detection-backend
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Vérifier les modèles

Assurez-vous que tous les fichiers de modèles sont présents :

- `model/xlmroberta_finetuned/`
- `model/prophet_model.json`
- `model/random_forest_meta_model.joblib`

### 4. Lancer l'API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

## 🚀 Utilisation de l'API

### Endpoints Disponibles

#### `POST /check` - Analyse avec modèle combiné (Recommandé)

Analyse une publication Facebook avec l'approche multi-modèles.

**Request:**

```json
{
  "facebook_url": "https://www.facebook.com/page/posts/123456789"
}
```

**Response:**

```json
{
  "source": "facebook",
  "facebook_url": "https://www.facebook.com/page/posts/123456789",
  "extracted_text": "Texte de la publication...",
  "extracted_date": "2025-01-15",
  "prediction": "FAKE",
  "confidence": 85.2,
  "model_details": {
    "final_probability": {
      "FAKE": 85.2,
      "REAL": 14.8
    },
    "scores": {
      "nlp_score": 0.8234,
      "temporal_score": 0.6789
    },
    "nlp_only": {
      "prediction": "FAKE",
      "confidence": 82.34
    }
  },
  "fact_check": {
    "status": "Vérification effectuée",
    "similarity_score": 0.75,
    "source_url": "https://source-fiable.com/article",
    "article_date": "2025-01-10"
  }
}
```

#### `POST /check_legacy` - Analyse NLP seule (Legacy)

Utilise uniquement le modèle XLM-RoBERTa (ancien système).

#### `GET /` - Documentation API

Informations sur les endpoints disponibles.

#### `GET /health` - Santé de l'API

Vérification du statut de l'API.

## 🧠 Détails des Modèles

### 1. Modèle NLP (XLM-RoBERTa)

- **Fonction** : Analyse sémantique du contenu textuel
- **Input** : Texte de la publication
- **Output** : Score de vraisemblance [0,1] où 0=FAKE, 1=REAL
- **Localisation** : `model/xlmroberta_finetuned/`

### 2. Modèle Temporel (Prophet)

- **Fonction** : Analyse des tendances temporelles des fake news
- **Input** : Date de publication
- **Output** : Score temporel [0,1] basé sur les patterns historiques
- **Localisation** : `model/prophet_model.json`

### 3. Méta-modèle (Random Forest)

- **Fonction** : Combine les scores NLP et temporel pour une prédiction finale
- **Input** : [score_nlp, score_temporel]
- **Output** : Prédiction finale FAKE/REAL avec probabilités
- **Localisation** : `model/random_forest_meta_model.joblib`

## 📊 Fonctionnalités Avancées

### Extraction Automatique

- **Facebook Scraping** : Extraction automatique du contenu et de la date
- **Gestion des erreurs** : Fallback graceful en cas d'échec
- **Formats multiples** : Support de différents formats d'URL Facebook

### Fact-checking

- **Recherche Web** : Vérification automatique via sources fiables
- **Score de similarité** : Comparaison avec articles vérifiés
- **Sources datées** : Prise en compte de la temporalité des sources

### API Robuste

- **Gestion d'erreurs** : Messages d'erreur détaillés
- **CORS configuré** : Support pour applications web frontend
- **Documentation automatique** : Swagger UI disponible sur `/docs`

## 🔍 Exemple d'Usage Complet

```python
import requests

# Configuration
api_url = "http://localhost:8000"
facebook_url = "https://www.facebook.com/page/posts/123456789"

# Requête
response = requests.post(
    f"{api_url}/check",
    json={"facebook_url": facebook_url}
)

result = response.json()

# Analyse des résultats
print(f"Prédiction: {result['prediction']}")
print(f"Confiance: {result['confidence']}%")
print(f"Score NLP: {result['model_details']['scores']['nlp_score']}")
print(f"Score Temporel: {result['model_details']['scores']['temporal_score']}")
```

## 🛡️ Considérations de Sécurité

- **Validation des entrées** : Vérification stricte des URLs Facebook
- **Limitation des ressources** : Protection contre les attaques par déni de service
- **Gestion des erreurs** : Pas d'exposition d'informations sensibles

## 📈 Performance et Optimisation

- **Cache des modèles** : Chargement unique au démarrage
- **Traitement parallèle** : Scores calculés de manière optimisée
- **Gestion mémoire** : Utilisation efficace des ressources GPU/CPU

## 🔧 Configuration Avancée

### Variables d'environnement

```bash
export FACEBOOK_SCRAPER_TIMEOUT=30
export MODEL_CACHE_SIZE=1000
export API_LOG_LEVEL=INFO
```

### Monitoring

- Logs détaillés pour le debugging
- Métriques de performance des modèles
- Tracking des erreurs et exceptions

## 🤝 Contribution

Pour contribuer au projet :

1. Fork le repository
2. Créer une branche feature (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commiter les changements (`git commit -am 'Ajout nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Maheni Ntsoa** - Développement principal
- **Équipe de recherche** - Entraînement des modèles

## 📞 Support

Pour toute question ou problème :

- Créer une issue sur GitHub
- Contact : [email de contact]

---

_Dernière mise à jour : Août 2025_
