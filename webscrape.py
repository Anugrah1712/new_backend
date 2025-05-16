import hashlib
import os
import pickle
import asyncio
from urllib.parse import urljoin, urlparse
from playwright.async_api import async_playwright
import google.generativeai as genai

# Gemini setup
genai.configure(api_key="AIzaSyD364sF7FOZgaW4ktkIcITe_7miCqjhs4k")
model = genai.GenerativeModel("gemini-1.5-flash")

# Cache file
CACHE_PATH = "scraped_cache.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        scraped_cache = pickle.load(f)
else:
    scraped_cache = {}

def hash_url(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()

def is_internal_link(base_url, link):
    try:
        parsed_base = urlparse(base_url)
        parsed_link = urlparse(link)
        return (parsed_base.netloc == parsed_link.netloc) or (parsed_link.netloc == "")
    except:
        return False

async def scrape_page(url, browser):
    url_hash = hash_url(url)
    if url_hash in scraped_cache:
        print(f"[CACHE] Using cached data for {url}")
        return scraped_cache[url_hash]

    print(f"[SCRAPE] Scraping {url}")
    page = await browser.new_page()
    try:
        await page.goto(url, timeout=60000)
        await page.wait_for_timeout(3000)

        # Extract tables
        tables = await page.query_selector_all("table")
        table_data_list = []
        for i, table in enumerate(tables, start=1):
            rows = await table.query_selector_all("tr")
            table_data = []
            for row in rows:
                cols = await row.query_selector_all("td")
                col_text = [await col.inner_text() for col in cols if col]
                if col_text:
                    table_data.append(col_text)
            if table_data:
                summary = await convert_table_to_sentences_gemini(table_data, i)
                table_data_list.append(summary)

        # Extract full text
        body = await page.query_selector("body")
        full_text = await body.inner_text() if body else ""

        # Extract FAQs
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
            print(f"[DEBUG] FAQ extraction failed for {url}: {e}")

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
        print("\n================= [SCRAPED PAGE SUMMARY] =================")
        print(f"URL              : {url}")
        print(f"Text Length      : {len(full_text.split())} words")
        print(f"Tables Found     : {len(table_data_list)}")
        print(f"FAQs Found       : {len(faqs)}")
        print("Text Preview     :")
        print(full_text.strip()[:300].replace("\n", " ") + "...")
        print("===========================================================\n")

        print(f"[SCRAPE DONE] {url}")
        return result

    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return None
    finally:
        await page.close()

async def get_internal_links(url, browser):
    page = await browser.new_page()
    await page.goto(url, timeout=60000)
    await page.wait_for_timeout(2000)

    anchors = await page.query_selector_all("a")
    links = set()
    for anchor in anchors:
        try:
            href = await anchor.get_attribute("href")
            if href and is_internal_link(url, href):
                absolute_url = urljoin(url, href.split("#")[0])
                links.add(absolute_url)
        except:
            continue
    await page.close()

    print(f"[INTERNAL LINKS] Found {len(links)} links in {url}")  # ✅ Move here
    return list(links)


async def scrape_with_playwright_recursive(main_url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        to_scrape = set()
        scraped = set()

        # Start with main page
        to_scrape.add(main_url)
        results = []

        while to_scrape:
            current_url = to_scrape.pop()
            if current_url in scraped:
                continue

            print(f"[VISIT] {current_url}")
            result = await scrape_page(current_url, browser)
            if result:
                results.append(result)
                scraped.add(current_url)

                # Find more links on this page
                try:
                    sub_links = await get_internal_links(current_url, browser)
                    for link in sub_links:
                        if link not in scraped:
                            to_scrape.add(link)
                    
                except Exception as e:
                    print(f"[LINKS] Failed to extract internal links from {current_url}: {e}")

        await browser.close()
        
        return results

async def convert_table_to_sentences_gemini(table_data, index):
    table_input = f"Table {index}:\n" + "\n".join([", ".join(row) for row in table_data])
    chat = model.start_chat(history=[
        {"role": "user", "parts": [
            "Convert table to descriptive sentences.\n"
            
        ]},
        {"role": "model", "parts": ["Please provide the table."]}
    ])
    response = await chat.send_message(table_input)
    return response.text

async def scrape_web_data(url):
    return await scrape_with_playwright_recursive(url)
