# fact_checker.py
# Module de vérification des faits pour la détection de fake news
# Dernière vérification des sites médias : Août 2025
# Note : Plusieurs sites médiatiques malgaches ne sont plus accessibles ou ont changé de nature

from googlesearch import search
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from get_article_date import get_article_date

model_sem = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

MEDIA_SITES = [
    # === SITES ACTIFS ET FIABLES (vérifiés en août 2025) ===
    "lexpress.mg",                    # ✅ Actif - L'Express de Madagascar
    "midi-madagasikara.mg",           # ✅ Actif - Midi Madagasikara
    "madagascar-tribune.com",         # ✅ Actif - Madagascar Tribune
    "malagasynews.com",              # ✅ Actif - Malagasy News
    "sobika.com",                    # ✅ Actif - Sobika
    "allafrica.com/madagascar",      # ✅ Actif - AllAfrica Madagascar
    "allafrica.com/madagascar",      # ✅ Actif - AllAfrica Madagascar
    "actu.orange.mg",                 # ✅ Actif - Actu Orange

    # === SITES INACTIFS/PROBLÉMATIQUES (commentés temporairement) ===
    # "rta.mg",                      # ❌ Site inaccessible (connexion échouée)
    # "tvm.mg",                      # ❌ Site inaccessible (connexion échouée)
    # "la-verite.mg",                # ❌ Site inaccessible (connexion échouée)
    # "newsmada.com",                # ❌ Erreur 406 (accès refusé par le serveur)
    # "lagazette-dgi.com",           # ❌ COMPROMIS - devenu un site de casino/paris
]

FACEBOOK_PAGES_FIABLES = [
    "24hMada",
    "NewsMada",
    "OrangeActusMadagascar",
    "MidiMadagasikara",
    "lexpress.mg",
    "MadagascarTribune",
    "LaVeriteMadagascar",
    "RTAMadagascar",
    "TVMMadagascar",
]

# Suppression des doublons
MEDIA_SITES = list(sorted(set(MEDIA_SITES)))

def get_google_links(query, num_results=5):
    try:
        return list(search(query, num_results=num_results))
    except:
        return []

def extract_text_from_url(url):
    """
    Extrait le texte principal d'une page web en se concentrant sur le contenu de l'article
    """
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Supprimer les éléments non pertinents
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
            element.decompose()
        
        # Chercher le contenu principal dans l'ordre de priorité
        content_selectors = [
            'article',  # Balise article HTML5
            'main',     # Contenu principal
            '[role="main"]',  # Rôle main
            '.content', '.article', '.post', '.entry',  # Classes communes
            '#content', '#article', '#post', '#main',   # IDs communes
            '.article-content', '.post-content', '.entry-content'  # Classes spécifiques
        ]
        
        article_content = None
        for selector in content_selectors:
            article_content = soup.select_one(selector)
            if article_content:
                break
        
        # Si aucun contenu spécifique trouvé, utiliser le body
        if not article_content:
            article_content = soup.find('body') or soup
        
        # Extraire et nettoyer le texte
        text = article_content.get_text()
        
        # Nettoyage avancé
        import re
        text = re.sub(r'\s+', ' ', text)  # Normaliser les espaces
        text = text.strip()
        
        # Filtrer les lignes trop courtes (navigation, menus)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        text = ' '.join(lines)
        
        return text
        
    except Exception as e:
        print(f"Erreur extraction de {url}: {e}")
        return ""

def compute_similarity(text1, text2):
    emb1 = model_sem.encode(text1, convert_to_tensor=True)
    emb2 = model_sem.encode(text2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb1, emb2).item()
    return round(score, 3)

def is_facebook_publication(url):
    return "facebook.com/" in url

def get_facebook_page_name(url):
    parts = url.split("/")
    try:
        idx = parts.index("facebook.com")
        page_name = parts[idx+1]
        if page_name in ["fr", "en"]:
            page_name = parts[idx+2] if len(parts) > idx+2 else ""
        return page_name.split("?")[0]
    except (ValueError, IndexError):
        return None

def is_date_valid(article_date_str, reference_date_str):
    try:
        article_date = datetime.strptime(article_date_str, "%Y-%m-%d")
        reference_date = datetime.strptime(reference_date_str, "%Y-%m-%d")
        if article_date.year != reference_date.year:
            return False
        delta = abs((article_date - reference_date).days)
        return delta <= 92
    except Exception:
        return False

def is_site_in_trusted_list(url):
    """
    Vérifie si une URL appartient à notre liste de sites médiatiques fiables
    """
    for site in MEDIA_SITES:
        if site in url:
            return True
    return False

def fact_check(text, date_reference, is_real_news_nlp):
    keywords = text.split()[:15]
    query = " ".join(keywords)
    
    # === NOUVELLE APPROCHE : Recherche globale puis filtrage ===
    # 1. Recherche globale sur Google (5 premiers résultats)
    all_urls = get_google_links(query, num_results=5)
    
    # 2. Filtrer les URLs qui correspondent à nos sites fiables
    trusted_urls = []
    other_urls = []
    
    for url in all_urls:
        if is_facebook_publication(url):
            page_name = get_facebook_page_name(url)
            if page_name and any(page_name.lower() == name.lower() for name in FACEBOOK_PAGES_FIABLES):
                trusted_urls.append(url)
            else:
                other_urls.append(url)
        elif is_site_in_trusted_list(url):
            trusted_urls.append(url)
        else:
            other_urls.append(url)
    
    # 3. Vérifier d'abord les sites fiables
    for url in trusted_urls:
        article_text = extract_text_from_url(url)
        score = compute_similarity(text, article_text)
        article_date = get_article_date(url)
        date_ok = article_date and is_date_valid(article_date, date_reference)
        
        # Seuil plus bas pour les sites fiables (amélioration de l'extraction)
        if is_real_news_nlp and score > 0.35 and date_ok:
            return {
                "status": "real_news", 
                "url": url, 
                "similarity": score, 
                "date": article_date,
                "source_type": "trusted_media"
            }
    
    # 4. Si rien trouvé dans les sites fiables, vérifier les autres (avec seuil plus élevé)
    for url in other_urls:
        if not is_facebook_publication(url):  # Éviter les Facebook non fiables
            article_text = extract_text_from_url(url)
            score = compute_similarity(text, article_text)
            article_date = get_article_date(url)
            date_ok = article_date and is_date_valid(article_date, date_reference)
            
            # Seuil plus élevé pour les sources non fiables
            if is_real_news_nlp and score > 0.6 and date_ok:
                return {
                    "status": "real_news", 
                    "url": url, 
                    "similarity": score, 
                    "date": article_date,
                    "source_type": "other_source"
                }
    
    return {
        "status": "not_confirmed", 
        "url": None, 
        "similarity": None, 
        "date": None,
        "source_type": None,
        "debug_info": {
            "total_urls_found": len(all_urls),
            "trusted_urls_found": len(trusted_urls),
            "other_urls_found": len(other_urls),
            "trusted_urls": trusted_urls[:3],  # Les 3 premiers pour debug
            "other_urls": other_urls[:3]      # Les 3 premiers pour debug
        }
    }