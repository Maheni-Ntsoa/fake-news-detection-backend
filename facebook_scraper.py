from apify_client import ApifyClient
from datetime import datetime
import requests
import re
from urllib.parse import urlparse
import json
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By  # Correction de 'Byx' en 'By'
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Optional, Tuple

def scrape_facebook_with_selenium(post_url: str, headless: bool = True) -> tuple[str | None, str | None]:
    """
    Méthode utilisant Selenium WebDriver pour scraper Facebook.
    Cette méthode simule un navigateur réel et peut contourner certaines restrictions.
    
    Prérequis: 
    - pip install selenium
    - Télécharger ChromeDriver et l'ajouter au PATH
    
    Args:
        post_url: L'URL de la publication Facebook
        headless: Si True, le navigateur s'exécute en mode invisible
    
    Returns:
        Un tuple (texte_publication, date_formatée)
    """
    driver = None
    try:
        print(f"🌐 Scraping avec Selenium : {post_url}")
        
        # Configuration de Chrome
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        # Initialiser le driver
        driver = webdriver.Chrome(options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        # Naviguer vers la page
        driver.get(post_url)
        
        # Attendre que la page se charge
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Attendre un peu plus pour que le contenu dynamique se charge
        time.sleep(3)
        
        # Essayer différents sélecteurs pour le texte du post
        text_selectors = [
            '[data-ad-preview="message"]',
            '[data-testid="post_message"]',
            '.userContent',
            '.text_exposed_root',
            '[role="article"] p',
            'div[dir="auto"]'
        ]
        
        post_text = None
        for selector in text_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    post_text = elements[0].text.strip()
                    if post_text and len(post_text) > 10:  # Vérifier que ce n'est pas juste un placeholder
                        break
            except:
                continue
        
        # Essayer de trouver la date
        date_selectors = [
            'abbr[data-utime]',
            '[data-testid="story-subtitle"] a',
            '.timestampContent',
            'time'
        ]
        
        formatted_date = None
        for selector in date_selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    date_element = elements[0]
                    # Essayer d'extraire la date depuis l'attribut data-utime (timestamp Unix)
                    timestamp = date_element.get_attribute('data-utime')
                    if timestamp:
                        dt_object = datetime.fromtimestamp(int(timestamp))
                        formatted_date = dt_object.strftime('%Y-%m-%d')
                        break
                    # Sinon essayer le texte
                    date_text = date_element.text or date_element.get_attribute('title')
                    if date_text:
                        # Utiliser la date actuelle comme fallback
                        formatted_date = datetime.now().strftime('%Y-%m-%d')
                        break
            except:
                continue
        
        if not formatted_date:
            formatted_date = datetime.now().strftime('%Y-%m-%d')
        
        print("✅ Scraping Selenium réussi")
        return post_text, formatted_date
        
    except Exception as e:
        print(f"❌ Erreur Selenium : {e}")
        return None, None
    finally:
        if driver:
            driver.quit()


def scrape_facebook_with_graph_api(post_id: str, access_token: str) -> tuple[str | None, str | None]:
    """
    Utilise l'API Graph de Facebook pour récupérer les données d'un post.
    GRATUIT mais nécessite un token d'accès Facebook.
    
    Pour obtenir un token d'accès :
    1. Aller sur https://developers.facebook.com/tools/explorer/
    2. Sélectionner votre app ou créer une nouvelle app
    3. Générer un token avec les permissions 'public_content'
    
    Args:
        post_id: L'ID du post Facebook (ex: "pageId_postId")
        access_token: Token d'accès Facebook
    
    Returns:
        Un tuple (texte_publication, date_formatée)
    """
    try:
        print(f"📊 Scraping avec Graph API : {post_id}")
        
        # URL de l'API Graph
        api_url = f"https://graph.facebook.com/v18.0/{post_id}"
        
        # Paramètres de la requête
        params = {
            'fields': 'message,created_time,story',
            'access_token': access_token
        }
        
        # Faire la requête
        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extraire le texte
        post_text = data.get('message') or data.get('story')
        
        # Extraire et formater la date
        created_time = data.get('created_time')
        formatted_date = None
        if created_time:
            dt_object = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
            formatted_date = dt_object.strftime('%Y-%m-%d')
        
        print("✅ Scraping Graph API réussi")
        return post_text, formatted_date
        
    except Exception as e:
        print(f"❌ Erreur Graph API : {e}")
        return None, None


def scrape_facebook_with_beautifulsoup(post_url: str) -> tuple[str | None, str | None]:
    """
    Méthode utilisant BeautifulSoup pour parser le HTML Facebook.
    Méthode simple mais peut être limitée par les mesures anti-scraping.
    
    Prérequis: pip install beautifulsoup4 lxml
    
    Args:
        post_url: L'URL de la publication Facebook
    
    Returns:
        Un tuple (texte_publication, date_formatée)
    """
    try:
        print(f"🍜 Scraping avec BeautifulSoup : {post_url}")
        
        # Headers pour simuler un navigateur mobile (souvent moins protégé)
        headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Essayer d'abord la version mobile de Facebook
        mobile_url = post_url.replace('www.facebook.com', 'm.facebook.com')
        
        response = requests.get(mobile_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Parser le HTML
        soup = BeautifulSoup(response.content, 'lxml')
        
        # Essayer différents sélecteurs pour le texte
        text_selectors = [
            'div[data-ft] span',
            '.story_body_container p',
            'div.msg span',
            '.userContent',
            'p',
            'span'
        ]
        
        post_text = None
        for selector in text_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if text and len(text) > 20:  # Prendre le texte le plus long
                    if not post_text or len(text) > len(post_text):
                        post_text = text
        
        # Nettoyer le texte
        if post_text:
            post_text = re.sub(r'\s+', ' ', post_text).strip()
        
        # Pour la date, utiliser la date actuelle comme fallback
        formatted_date = datetime.now().strftime('%Y-%m-%d')
        
        print("✅ Scraping BeautifulSoup réussi")
        return post_text, formatted_date
        
    except Exception as e:
        print(f"❌ Erreur BeautifulSoup : {e}")
        return None, None


def scrape_facebook_with_scrapy_splash(post_url: str, splash_url: str = "http://localhost:8050") -> tuple[str | None, str | None]:
    """
    Utilise Scrapy-Splash pour le rendu JavaScript et le scraping.
    Splash est un service de rendu JavaScript headless gratuit.
    
    Installation:
    1. pip install scrapy-splash requests
    2. docker run -p 8050:8050 scrapinghub/splash
    
    Args:
        post_url: L'URL de la publication Facebook
        splash_url: URL du service Splash (par défaut localhost:8050)
    
    Returns:
        Un tuple (texte_publication, date_formatée)
    """
    try:
        print(f"💦 Scraping avec Splash : {post_url}")
        
        # Paramètres pour Splash
        splash_params = {
            'url': post_url,
            'html': 1,
            'png': 0,
            'wait': 3,
            'lua_source': '''
                function main(splash, args)
                    splash:go(args.url)
                    splash:wait(3)
                    return splash:html()
                end
            '''
        }
        
        # Faire la requête à Splash
        response = requests.get(f"{splash_url}/execute", params=splash_params, timeout=30)
        response.raise_for_status()
        
        # Parser le HTML rendu
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Essayer d'extraire le texte du post
        text_selectors = [
            '[data-testid="post_message"]',
            '.userContent',
            '[role="article"] p',
            'div[dir="auto"]'
        ]
        
        post_text = None
        for selector in text_selectors:
            elements = soup.select(selector)
            if elements:
                post_text = elements[0].get_text(strip=True)
                if post_text:
                    break
        
        # Date actuelle comme fallback
        formatted_date = datetime.now().strftime('%Y-%m-%d')
        
        print("✅ Scraping Splash réussi")
        return post_text, formatted_date
        
    except Exception as e:
        print(f"❌ Erreur Splash : {e}")
        return None, None


def extract_post_id_from_url(post_url: str) -> str | None:
    """
    Extrait l'ID du post depuis une URL Facebook.
    Utile pour l'API Graph.
    
    Args:
        post_url: URL de la publication Facebook
    
    Returns:
        L'ID du post ou None si non trouvé
    """
    try:
        # Patterns pour extraire l'ID du post
        patterns = [
            r'/posts/(\d+)',
            r'/photos/[^/]+/(\d+)',
            r'story_fbid=(\d+)',
            r'fbid=(\d+)',
            r'/(\d+)/posts/(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, post_url)
            if match:
                if len(match.groups()) == 2:
                    return f"{match.group(1)}_{match.group(2)}"
                else:
                    return match.group(1)
        
        return None
    except Exception:
        return None
    
def scrape_facebook_fallback(
    post_url: str,
    access_token: Optional[str] = None,
    splash_url: str = "http://localhost:8050",
    headless: bool = True
) -> Tuple[str | None, str | None]:
    """
    Tente plusieurs méthodes de scraping jusqu'à ce que l'une réussisse.

    Args:
        post_url: L'URL de la publication Facebook
        access_token: Token pour la Graph API (optionnel)
        splash_url: URL de Splash (si utilisé)
        headless: Mode headless pour Selenium

    Returns:
        Tuple (texte_publication, date_formatée)
    """
    print("🚀 Démarrage du scraping multi-stratégies")

    # 1. Graph API (si token fourni)
    if access_token:
        post_id = extract_post_id_from_url(post_url)
        if post_id:
            text, date = scrape_facebook_with_graph_api(post_id, access_token)
            if text:
                return text, date

    # 2. Selenium
    text, date = scrape_facebook_with_selenium(post_url, headless=headless)
    if text:
        return text, date

    # 3. BeautifulSoup
    text, date = scrape_facebook_with_beautifulsoup(post_url)
    if text:
        return text, date

    # 4. Scrapy Splash
    text, date = scrape_facebook_with_scrapy_splash(post_url, splash_url=splash_url)
    if text:
        return text, date

    print("❌ Aucune méthode n'a réussi à extraire les données.")
    return None, None
