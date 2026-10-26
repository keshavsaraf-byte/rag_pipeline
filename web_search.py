
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS


def search_web(query, max_results=3):
    results = DDGS().text(
        query,
        max_results=max_results
    )

    web_pages = []

    for result in results:
        web_pages.append({
            "title": result.get("title", ""),
            "snippet": result.get("body", ""),
            "url": result.get("href", "")
        })

    return web_pages


def fetch_webpage(url):
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for element in soup([
        "script", "style", "nav", "footer",
        "header", "noscript", "iframe"
    ]):
        element.decompose()

    return soup.get_text(separator=" ", strip=True)


def search_and_fetch(query, max_results=3):
    search_results = search_web(query, max_results)
    web_documents = []

    for result in search_results:
        url = result["url"]

        if not url.startswith(("https://", "http://")):
            continue

        try:
            content = fetch_webpage(url)

            if not content.strip():
                continue

            web_documents.append({
                "title": result["title"],
                "url": url,
                "content": content[:10000]
            })

        except requests.RequestException as error:
            print(f"Could not fetch {url}: {error}")

    return web_documents


if __name__ == "__main__":
    query = input("Enter your search query: ")
    documents = search_and_fetch(query)

    for doc in documents:
        print("\nTitle:", doc["title"])
        print("URL:", doc["url"])
        print("Content:", doc["content"][:500])
