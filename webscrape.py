# webscrape.py

import os
import asyncio
import pickle
import hashlib
from playwright.async_api import async_playwright
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")

# Pickle file paths
WEB_SCRAPE_PICKLE = "scraped_data.pkl"
LINKS_HASH_FILE = "links_hash.pkl"

# Gemini API Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config=generation_config,
)

async def convert_table_to_sentences_gemini(tables):
    """Converts tables into descriptive sentences using Gemini API."""
    print("\n[INFO] Sending tables to Gemini for conversion to sentences...\n")

    if not tables:
        return "No tables found on the page."

    table_input = "\n\n".join(
        [f"Table {i}:\n" + "\n".join([" | ".join(row) for row in data]) for i, data in enumerate(tables, 1)]
    )

    chat_session = model.start_chat()
    response = chat_session.send_message(f"Convert these tables to descriptive sentences:\n{table_input}")

    return response.text if response else "No response from Gemini API"

async def scrape_web_data(links=None):
    """Scrapes web data from given links and caches results. Uses cache if no new links are provided."""
    
    # If no new links are provided, load from cache
    if not links and os.path.exists(WEB_SCRAPE_PICKLE):
        with open(WEB_SCRAPE_PICKLE, "rb") as f:
            cached_text = pickle.load(f)
            print("✅ No new links provided. Loaded cached data!")
            return cached_text  # ✅ Ensure it returns a string, not a list

    # Compute a hash for the current set of links
    new_links_str = ",".join(links) if links else ""
    new_hash = hashlib.md5(new_links_str.encode()).hexdigest()

    # Check if the previous hash exists and matches
    if os.path.exists(LINKS_HASH_FILE):
        with open(LINKS_HASH_FILE, "rb") as f:
            old_hash = pickle.load(f)

        if new_hash == old_hash and os.path.exists(WEB_SCRAPE_PICKLE):
            with open(WEB_SCRAPE_PICKLE, "rb") as f:
                cached_text = pickle.load(f)
                print("✅ No changes detected in links. Loaded cached data!")
                return cached_text  # ✅ Return as string

    print("[INFO] Starting web scraping...\n")
    scraped_text = ""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Run in headless mode for performance
        page = await browser.new_page()

        for link in links:
            print(f"[INFO] Navigating to {link}\n")
            try:
                await page.goto(link, timeout=120000, wait_until="domcontentloaded")

                # Extract Full Page Text
                body_text = await page.evaluate("document.body.innerText")
                print("[INFO] Extracting full page content...\n")

                # Extract Tables
                tables = await extract_tables(page)
                print(tables)
                # Convert tables to descriptive sentences
                table_sentences = await convert_table_to_sentences_gemini(tables)
                print(table_sentences)
                # Extract FAQs
                faqs = await extract_faqs(page)
                faq_text = "\n\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in faqs])

                # Concatenate all extracted content into a single string
                scraped_text += f"\n\n--- Page Content from {link} ---\n{body_text[:1000]}\n\nTables:\n{table_sentences}\n\nFAQs:\n{faq_text}\n\n"
                print(scraped_text)
            except Exception as e:
                print(f"[ERROR] Failed to scrape {link}: {e}")

        await browser.close()

    # Save new scraped data to cache
    with open(WEB_SCRAPE_PICKLE, "wb") as f:
        pickle.dump(scraped_text, f)
        print("💾 New scraped data saved to cache!")

    # Update hash file
    with open(LINKS_HASH_FILE, "wb") as f:
        pickle.dump(new_hash, f)

    return scraped_text  # ✅ Return as a single string

async def extract_tables(page):
    """Extracts tables from the webpage and returns as a formatted string."""
    print("\n[INFO] Extracting tables...\n")
    tables = await page.query_selector_all("table")
    table_data = []

    for i, table in enumerate(tables, 1):
        rows = await table.query_selector_all("tr")
        table_content = []

        for row in rows:
            columns = await row.query_selector_all("td, th")  # Include headers
            column_text = [await column.inner_text() for column in columns]
            column_text = [text.strip() for text in column_text if text.strip()]
            if column_text:
                table_content.append(column_text)

        if table_content:
            table_data.append(table_content)

    # Convert table data to a readable string
    if table_data:
        return "\n\n".join(
            [f"Table {i}:\n" + "\n".join([" | ".join(row) for row in table]) for i, table in enumerate(table_data, 1)]
        )
    
    return "No tables found."

async def extract_faqs(page):
    """Extracts FAQs from the page and returns as a list of dictionaries."""
    print("\n[INFO] Extracting FAQs...\n")
    faq_container = await page.query_selector(".faqs.aem-GridColumn.aem-GridColumn--default--12")

    if not faq_container:
        return []

    all_faqs = []

    for _ in range(10):  # Avoid infinite loop
        show_more_button = await faq_container.query_selector(".accordion_toggle_show-more")
        if show_more_button and await show_more_button.is_visible():
            await show_more_button.click()
            await page.wait_for_timeout(1000)
        else:
            break

    toggle_buttons = await faq_container.query_selector_all(".accordion_toggle, .accordion_row")

    for button in toggle_buttons:
        try:
            await button.click()
            await page.wait_for_timeout(1000)
            expanded_content = await faq_container.query_selector_all(".accordion_body, .accordionbody_links, .aem-rte-content")
            for content in expanded_content:
                text = await content.inner_text()
                text = text.strip()
                if text and text not in [faq['answer'] for faq in all_faqs]:
                    question = await button.inner_text()
                    question = question.strip()
                    if question:
                        all_faqs.append({"question": question, "answer": text})
        except Exception as e:
            print(f"[ERROR] Error extracting FAQs: {e}")

    return all_faqs

# Run the scraper
async def main():
    scraped_data = await scrape_web_data()  
    print("\nScraped Data:\n", scraped_data)

if __name__ == "__main__":
    asyncio.run(main())
