from urlextract import URLExtract
from readability import Document
from bs4 import BeautifulSoup

import requests


def fetch_urls(text, prefix='http://'):
    extractor = URLExtract()
    urls = extractor.find_urls(text)
    try:
        for url in urls:
            fixed_url = url if prefix in url else f'{prefix}{url}'
            page_content = get_page_content(fixed_url)
            if not page_content:
                continue

            links = extract_rss_links(page_content)
            if len(links) > 0:
                rss_feed_content = get_page_content(links[0])
                if not rss_feed_content:
                    continue
                data = summarize_page(rss_feed_content)
            else:
                # no RSS feed found. Just summarize webpage.
                data = summarize_page(page_content)

            text = text.replace(url, f' : {data}')
    except Exception:
        pass
    return text


def get_page_content(url):
    res = requests.get(url)
    if res.status_code == 200:
        return res.content
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