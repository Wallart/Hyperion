from urlextract import URLExtract
from readability import Document
from bs4 import BeautifulSoup

import requests


def fetch_urls(text, prefix='http://'):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    for url in urls:
        fixed_url = url
        if prefix not in url:
            fixed_url = prefix + url

        response = requests.get(fixed_url)
        doc = Document(response.content)

        sumup = f': {doc.title()} {doc.summary()}'
        clean_sumup = BeautifulSoup(sumup, 'lxml').text
        text = text.replace(url, clean_sumup)

    return text
