# Webscrape

import hashlib
import os
import pickle
import asyncio
from playwright.async_api import async_playwright
import google.generativeai as genai


# Configure Gemini
genai.configure(api_key="AIzaSyD364sF7FOZgaW4ktkIcITe_7miCqjhs4k")
model = genai.GenerativeModel("gemini-1.5-flash")

# Cache path
CACHE_PATH = "scraped_cache.pkl"

# Load or create cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        scraped_cache = pickle.load(f)
else:
    scraped_cache = {}

def hash_url(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

async def scrape_with_playwright(url):
    url_hash = hash_url(url)
    if url_hash in scraped_cache:
        print(f"[CACHE] Using cached data for {url}")
        return scraped_cache[url_hash]

    print(f"[SCRAPE] Scraping {url}...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=60000)
        await page.wait_for_timeout(3000)

        # Extract tables
        tables = await page.query_selector_all("table")
        table_data_list = []
        print(f"[DEBUG] Found {len(tables)} table(s).")

        for i, table in enumerate(tables, start=1):
            rows = await table.query_selector_all("tr")
            table_data = []
            for row in rows:
                cols = await row.query_selector_all("td")
                col_text = [await col.inner_text() for col in cols if col]
                if col_text:
                    table_data.append(col_text)
            if table_data:
                print(f"[DEBUG] Table {i} raw data:\n{table_data}")
                summary = convert_table_to_sentences_gemini(table_data, i)
                print(f"[DEBUG] Table {i} summary:\n{summary}\n")
                table_data_list.append(summary)

        # Extract full body text
        body = await page.query_selector("body")
        full_text = await body.inner_text() if body else ""
        print(f"[DEBUG] Full page text (first 1000 chars):\n{full_text[:1000]}\n")

        # Expand and extract FAQs
        faqs = []
        try:
            faq_container = await page.query_selector(".faqs.aem-GridColumn.aem-GridColumn--default--12")
            if faq_container:
                while True:
                    try:
                        button = await faq_container.query_selector(".accordion_toggle_show-more")
                        if button and await button.is_visible():
                            await button.click()
                            await page.wait_for_timeout(1000)
                        else:
                            break
                    except:
                        break

                buttons = await faq_container.query_selector_all(".accordion_toggle, .accordion_row")
                for btn in buttons:
                    try:
                        await btn.click()
                        await page.wait_for_timeout(500)
                        expanded = await faq_container.query_selector_all(".accordion_body, .accordionbody_links, .aem-rte-content")
                        for content in expanded:
                            answer = await content.inner_text()
                            question = await btn.inner_text()
                            if answer.strip() and question.strip():
                                faqs.append({"question": question.strip(), "answer": answer.strip()})
                    except:
                        continue
        except Exception as e:
            print(f"[DEBUG] FAQ extraction failed: {e}")

        if faqs:
            print(f"[DEBUG] Extracted {len(faqs)} FAQ(s):")
            for faq in faqs:
                print(f"Q: {faq['question']}\nA: {faq['answer']}\n")

        await browser.close()

        # Combine everything
        result = {
            "url": url,
            "full_text": full_text.strip(),
            "table_summaries": table_data_list,
            "faqs": faqs,
        }

        # Cache result
        scraped_cache[url_hash] = result
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(scraped_cache, f)

        print(f"[SCRAPE DONE] Returning scraped data for {url}\n")
        return result


def convert_table_to_sentences_gemini(table_data, index):
    table_input = f"Table {index}:\n" + "\n".join([", ".join(row) for row in table_data])
    chat = model.start_chat(history=[
        {"role": "user", "parts": [
            "Convert table to descriptive sentences. For example:\n"
            "The following information is for the customers under the age of 60 with a special period.● "
            "For customers with a tenure of 18 months... (rest of example)"
        ]},
        {"role": "model", "parts": ["Please provide the table."]}
    ])
    response = chat.send_message(table_input)
    return response.text

async def scrape_web_data(url):
    return await scrape_with_playwright(url)


