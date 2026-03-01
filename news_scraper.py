
import requests
from bs4 import BeautifulSoup
import streamlit as st

# --- Constants ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# --- Helper for error handling ---
def _handle_request_error(e, source):
    # Using st.warning to show a non-blocking message
    st.warning(f"Không thể tải tin tức từ {source}: {e}")
    return []

# --- Scraper for CafeF ---
@st.cache_data(ttl=900) # Cache for 15 minutes
def get_cafef_news(max_items=7):
    """Scrapes the latest stock market news from CafeF.vn."""
    URL = "https://cafef.vn/timeline/31/trang-1.chn"
    articles = []
    try:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        news_list = soup.find('ul', class_='tl-stream')
        if not news_list:
            return []
        for item in news_list.find_all('li', limit=max_items):
            title_element = item.find('h3', class_='title')
            if title_element and title_element.a:
                title = title_element.a.get_text(strip=True)
                link = "https://cafef.vn" + title_element.a['href']
                articles.append({'title': title, 'link': link})
    except requests.exceptions.RequestException as e:
        return _handle_request_error(e, "CafeF")
    except Exception: # Broad exception for parsing errors
        return []
    return articles

# --- Scraper for Vietstock ---
@st.cache_data(ttl=900) # Cache for 15 minutes
def get_vietstock_news(max_items=7):
    """Scrapes the latest financial news from Vietstock.vn."""
    URL = "https://vietstock.vn/kinh-te.htm"
    articles = []
    try:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        news_list = soup.find_all('div', class_='article-content', limit=max_items)
        if not news_list:
            return []
        for item in news_list:
            title_element = item.find('a', class_='channel-title')
            if title_element:
                title = title_element.get_text(strip=True)
                link = title_element['href']
                # Ensure link is absolute
                if not link.startswith('http'):
                    link = "https://vietstock.vn" + link
                articles.append({'title': title, 'link': link})
    except requests.exceptions.RequestException as e:
        return _handle_request_error(e, "Vietstock")
    except Exception: # Broad exception for parsing errors
        return []
    return articles

# --- Scraper for NDH.vn (Nhip Cau Dau Tu) ---
@st.cache_data(ttl=900) # Cache for 15 minutes
def get_ndh_news(max_items=7):
    """Scrapes the latest business news from NDH.vn."""
    URL = "https://ndh.vn/doanh-nghiep"
    articles = []
    try:
        response = requests.get(URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        news_list = soup.find_all('div', class_='list-news__item', limit=max_items)
        if not news_list:
            return []
        for item in news_list:
            title_element = item.find('a', class_='list-news__title')
            if title_element:
                title = title_element.get_text(strip=True)
                link = title_element['href']
                # Ensure link is absolute
                if not link.startswith('http'):
                    link = "https://ndh.vn" + link
                articles.append({'title': title, 'link': link})
    except requests.exceptions.RequestException as e:
        return _handle_request_error(e, "NDH.vn")
    except Exception: # Broad exception for parsing errors
        return []
    return articles
