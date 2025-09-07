import requests

NEWSAPI_KEY = "e4d442912a3c4aa1a6cfa3d2e2e3e80b"

def verify_with_newsapi(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={NEWSAPI_KEY}"
    response = requests.get(url).json()
    articles = response.get("articles", [])
    
    if not articles:
        return None  # no match found
    
    results = []
    for art in articles[:3]:  # limit to top 3
        results.append({
            "title": art["title"],
            "source": art["source"]["name"],
            "url": art["url"],
            "description": art.get("description", "")
        })
    return results
