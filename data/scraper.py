import json
import requests
from bs4 import BeautifulSoup
from utils.directories import PRETRAINING_PAGES_DIR, PRETRAINING_RAW_DATASET_DIR

headers = {"User-Agent": "llm/1.0"}


def scrape(topic):
    print(f"scraping {topic}")
    html = requests.get(
        f'https://en.wikipedia.org/wiki/{topic.replace(" ", "_")}', headers=headers
    ).text

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
    with open(PRETRAINING_RAW_DATASET_DIR, "w") as f:
        f.write("".join(extracted_text))


if __name__ == "__main__":

    extracted_text = []

    with open(PRETRAINING_PAGES_DIR, "r") as f:
        pages = json.load(f)
        for topic in pages:
            extracted_text.append(scrape(topic))

    save(extracted_text)
