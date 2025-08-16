#Scrapes the website provided, if scraping is not possible, then user types in job description in text field

#Need to fix the web scraping, scrapes unnecessary info
import requests
from bs4 import BeautifulSoup
def scrape_job(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    html_content = soup.prettify()
    if response.status_code in [403, 503]:
        return False
    if "access denied" in response.text.lower() or "request blocked" in response.text.lower():
        return False
    text_blocks = set()
    soup = BeautifulSoup(response.text, "html.parser")
    # KEYWORDS = [
    #         "responsibilities", "requirements", "qualifications",
    #         "preferred", "skills", "what you'll do", "who you are"
    #     ]

    BLOCK_IGNORE = [
            "related jobs", "more jobs", "similar jobs", "other jobs"
        ]

    for tag in soup.find_all(["div", "section", "article"]):
        block_text = tag.get_text(separator=" ", strip=True).lower()
        if any(phrase in block_text for phrase in BLOCK_IGNORE):
                continue
        #if any(keyword in block_text for keyword in KEYWORDS):
        text_blocks.add(block_text)
    if text_blocks:
        html_content = "\n\n".join(text_blocks)
        return html_content
    else:
        return False