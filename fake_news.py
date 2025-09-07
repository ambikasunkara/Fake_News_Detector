import requests
import pandas as pd
from bs4 import BeautifulSoup

API_KEY = "e4d442912a3c4aa1a6cfa3d2e2e3e80b"  # replace with your key
URL = "https://newsapi.org/v2/top-headlines"

params = {
    "country": "in",       # change to your country code (e.g., "us", "gb")
    "language": "en",
    "pageSize": 10,        # number of articles
    "apiKey": API_KEY
}

response = requests.get(URL, params=params)
data = response.json()

articles = []
for article in data["articles"]:
    title = article["title"]
    desc = article["description"]
    url = article["url"]

    # Fetch full article content
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.content, "html.parser")
        paragraphs = soup.find_all("p")
        full_text = " ".join([p.get_text() for p in paragraphs])
    except:
        full_text = desc if desc else title

    articles.append({"title": title, "url": url, "content": full_text})

# Save to CSV
df = pd.DataFrame(articles)
df.to_csv("data/LiveNews.csv", index=False)
print("[INFO] Saved latest news to data/LiveNews.csv")
