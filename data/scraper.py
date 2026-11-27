"""Gather the dataset by scraping random Wikipedia pages."""

import time
import json
import requests
from bs4 import BeautifulSoup
from utils.directories import PRETRAINING_PAGES_DIR, PRETRAINING_RAW_DATASET_DIR

# Wikipedia API expects this header
headers = {"User-Agent": "minilm/1.0"}


def fetch(url, retries=5, wait=120):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers)
            if response.status_code != 429:
                return response.text
        except requests.exceptions.ConnectionError:
            pass
        print(f"failed, waiting {wait}s (attempt {attempt}/{retries})")
        time.sleep(wait)
    return None


def scrape(topic):
    """Scrape the text from the given Wikipedia page."""
    print(f"scraping {topic}")
    html = fetch(f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}')
    if html is None:
        print(f"skipped {topic}")
        return ""

    # ge the raw html
    body = BeautifulSoup(html, "html.parser").select_one("div.mw-parser-output")

    if body is None:
        print(f"no page found for {topic}, skipping")
        return ""

    for tag in body.select(
        "sup.reference, .reflist, .navbox, table, .mw-editsection, style, .hatnote, .thumb"
    ):
        tag.decompose()

    text = "\n\n".join(
        [p.get_text(" ", strip=True) for p in body.find_all("p", recursive=True)]
    )

    return text


def save(extracted_text):
    """Save the scraped text."""
    with open(PRETRAINING_RAW_DATASET_DIR, "w", encoding="utf-8") as f:
        f.write("".join(extracted_text))


if __name__ == "__main__":

    extracted_text = []

    with open(PRETRAINING_PAGES_DIR, "r", encoding="utf-8") as f:
        pages = json.load(f)
        for topic in pages:
            extracted_text.append(scrape(topic))

    save(extracted_text)
