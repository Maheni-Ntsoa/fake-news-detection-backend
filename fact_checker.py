# fact_checker.py
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
from get_article_date import get_article_date

model_sem = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

MEDIA_SITES = [
    # Extraits de https://www.newsmedialists.com/newspapers/madagascar
    "lexpress.mg",
    "midi-madagasikara.mg",
    "madagascar-tribune.com",
    "rta.mg",
    "tvm.mg",
    "malagasynews.com",
    "la-verite.mg",
    "lagazette-dgi.com",
    "sobika.com",
    "newsmada.com",
    "orange.mg",
    "allafrica.com/madagascar",
    # Extraits de https://www.journaux.info/madagascar
    "lexpress.mg",
    "midi-madagasikara.mg",
    "madagascar-tribune.com",
    "rta.mg",
    "tvm.mg",
    "la-verite.mg",
    "lagazette-dgi.com",
    "sobika.com",
    "newsmada.com",
    "malagasynews.com",
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
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except:
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

def fact_check(text, date_reference, is_real_news_nlp):
    keywords = text.split()[:15]
    query = " ".join(keywords)
    site_filters = " OR ".join([f"site:{site}" for site in MEDIA_SITES])
    full_query = f"{query} {site_filters}"

    urls = get_google_links(full_query)

    for url in urls:
        article_text = extract_text_from_url(url)
        score = compute_similarity(text, article_text)
        article_date = get_article_date(url)
        date_ok = article_date and is_date_valid(article_date, date_reference)
        if is_real_news_nlp and score > 0.4 and date_ok:
            return {"status": "real_news", "url": url, "similarity": score, "date": article_date}

    # Recherche globale si rien trouvé
    global_urls = get_google_links(query)
    for url in global_urls:
        if is_facebook_publication(url):
            page_name = get_facebook_page_name(url)
            if page_name and any(page_name.lower() == name.lower() for name in FACEBOOK_PAGES_FIABLES):
                article_text = extract_text_from_url(url)
                score = compute_similarity(text, article_text)
                article_date = get_article_date(url)
                date_ok = article_date and is_date_valid(article_date, date_reference)
                if is_real_news_nlp and score > 0.4 and date_ok:
                    return {"status": "real_news", "url": url, "similarity": score, "date": article_date}
        else:
            article_text = extract_text_from_url(url)
            score = compute_similarity(text, article_text)
            article_date = get_article_date(url)
            date_ok = article_date and is_date_valid(article_date, date_reference)
            if is_real_news_nlp and score > 0.4 and date_ok:
                return {"status": "real_news", "url": url, "similarity": score, "date": article_date}

    return {"status": "not_confirmed", "url": None, "similarity": None, "date": None}