import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

def load_url(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    text = soup.get_text(separator="\n")
    # Optionally convert to Markdown here (stub)
    metadata = {"source": url}
    return [Document(page_content=text, metadata=metadata)] 