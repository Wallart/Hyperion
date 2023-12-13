from bs4 import BeautifulSoup
from readability import Document
from urlextract import URLExtract
from hyperion.utils import ProjectLogger

import requests


def load_url(url):
    page_content = get_page_content(url)
    if not page_content:
        return False

    links = extract_rss_links(page_content)
    if len(links) == 0:
        # no RSS feed found. Just summarize webpage.
        return summarize_page(page_content)
    else:
        rss_feed_content = get_page_content(links[0])
        if not rss_feed_content:
            # no RSS feed found. Just summarize webpage.
            return summarize_page(page_content)
        return summarize_page(rss_feed_content)


def fetch_urls(text):
    prefixes = ('http://', 'https://')
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    try:
        for url in urls:
            fixed_url = f'{prefixes[0]}{url}' if (prefixes[0] not in url and prefixes[1] not in url) else url
            data = load_url(fixed_url)
            if data is False:
                continue
            text = text.replace(url, f': {data}')
    except Exception as e:
        ProjectLogger().warning(e)
    return text


def get_page_content(url):
    # Try to bypass antibot filters
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    }
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return res.content
    else:
        ProjectLogger().warning(res.reason)
    return False


def summarize_page(page_content):
    doc = Document(page_content)
    cleaned_summary = BeautifulSoup(f'{doc.title()} {doc.summary()}', 'lxml').text
    return cleaned_summary


def extract_rss_links(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')

    rss_link_tags = soup.find_all('link', type='application/rss+xml')
    atom_link_tags = soup.find_all('link', type='application/atom+xml')

    rss_links = [rss_link.get('href', '') for rss_link in rss_link_tags + atom_link_tags]
    return rss_links


if __name__ == '__main__':
    res = fetch_urls('lemonde.fr')
    print(res)