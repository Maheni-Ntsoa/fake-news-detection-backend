import requests
from bs4 import BeautifulSoup, Tag
import re

def get_article_date(url):
    """
    Tente d'extraire la date de publication d'un article web (format YYYY-MM-DD).
    Retourne None si la date n'est pas trouvée.
    """
    try:
        response = requests.get(url, timeout=8)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # 1. Cherche les balises meta courantes
        meta_date = soup.find("meta", {"property": "article:published_time"})
        if meta_date and isinstance(meta_date, Tag) and meta_date.get("content"):
            date_str = str(meta_date.get("content", ""))[:10]
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                return date_str
        
        meta_date2 = soup.find("meta", {"name": "pubdate"})
        if meta_date2 and isinstance(meta_date2, Tag) and meta_date2.get("content"):
            date_str = str(meta_date2.get("content", ""))[:10]
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                return date_str

        # 2. Cherche dans les balises time
        time_tag = soup.find("time")
        if time_tag and isinstance(time_tag, Tag) and time_tag.get("datetime"):
            date_str = str(time_tag.get("datetime", ""))[:10]
            if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
                return date_str

        # 3. Cherche un motif de date dans le texte
        match = re.search(r"(\d{4}-\d{2}-\d{2})", soup.text)
        if match:
            return match.group(1)
        
        return None
    except Exception as e:
        return None