import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import time
from urllib.parse import urljoin, urlparse
import streamlit as st

class WebSearcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search_duckduckgo(self, query, max_results=3):
        try:
            with DDGS() as ddgs:
                results = []
                for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'url': result.get('href', ''),
                        'snippet': result.get('body', '')
                    })
                return results
        except Exception as e:
            st.error(f"DuckDuckGo keresési hiba: {e}")
            return []

    def extract_content(self, url, max_chars=1000):
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main', 'article'])
            
            if main_content:
                text = main_content.get_text(strip=True, separator=' ')
            else:
                text = soup.get_text(strip=True, separator=' ')
            
            text = ' '.join(text.split())[:max_chars]
            return text
            
        except Exception as e:
            return f"Nem sikerült betölteni a tartalmat: {e}"

    def search_and_extract(self, query, max_results=3):
        search_results = self.search_duckduckgo(query, max_results)
        
        enhanced_results = []
        for result in search_results:
            content = self.extract_content(result['url'])
            enhanced_results.append({
                'title': result['title'],
                'url': result['url'],
                'snippet': result['snippet'],
                'content': content
            })
            time.sleep(0.5)
        
        return enhanced_results