import asyncio
import hashlib
import os
import pickle

from playwright.async_api import async_playwright
import google.generativeai as genai

genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-8b",
    generation_config={
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    },
)

os.makedirs("raw_structured_dumps", exist_ok=True)


def create_table_prompt(content):
    return (
        "You are analyzing a web page with one or more interest rate tables related to Fixed Deposits (FDs). "
        "For EACH table in the content below:\n"
        "- Mention the table's heading/title or any label that identifies the table.\n"
        "- Interpret all rows and columns precisely.\n"
        "- Clearly explain what each column means.\n"
        "- Summarize each row’s interest rate for all payout options.\n"
        "- Highlight the highest available rate and the corresponding tenure/payout.\n"
        "- Do NOT compare across tables. Each table should be explained independently.\n"
        "- Include any eligibility criteria near the tables if applicable.\n\n"
        "Here is the content:\n\n" + content
    )


def create_faq_prompt(content):
    return (
        "From the content below, extract up to 20 Frequently Asked Questions (FAQs). "
        "Include both questions found in the content and logical questions a user might ask. Format:\n"
        "Q: <question>\nA: <answer>\n\n"
        "Content:\n\n" + content
    )


def save_raw_content(link, content):
    hash_key = hashlib.md5(link.encode()).hexdigest()
    path = os.path.join("raw_structured_dumps", f"structured_{hash_key}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Saved structured content to {path}")


async def extract_structured_content(page):
    tables = await page.query_selector_all("table")
    content = ""
    for index, table in enumerate(tables, start=1):
        heading = f"\nTable {index}:\n"
        heading_element = await table.evaluate_handle(
            "(table) => table.previousElementSibling?.innerText || ''"
        )
        heading_text = await heading_element.json_value()
        heading += f"{heading_text.strip()}\n" if heading_text else ""
        rows = await table.query_selector_all("tr")
        for row in rows:
            cells = await row.query_selector_all("th, td")
            row_data = []
            for cell in cells:
                text = await cell.inner_text()
                row_data.append(text.strip())
            content += ", ".join(row_data) + "\n"
        content += "\n"

    body_text = await page.inner_text("body")
    content += "\nFULL PAGE TEXT:\n" + body_text
    return content


async def fetch_or_cache_data(link):
    hash_key = hashlib.md5(link.encode()).hexdigest()
    cache_file = f"scraped_{hash_key}.pkl"
    hash_file = f"hash_{hash_key}.pkl"

    if os.path.exists(hash_file):
        with open(hash_file, "rb") as f:
            old_hash = pickle.load(f)
        if old_hash == hash_key and os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                print(f"✅ [CACHE] Using cached data for: {link}")
                return pickle.load(f)

    print(f"🌐 Scraping fresh data for: {link}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--start-maximized",
            ]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            locale="en-US",
            viewport={"width": 1280, "height": 800}
        )

        page = await context.new_page()

        try:
            await page.goto(link, timeout=10000)
            await page.wait_for_load_state("networkidle")

            # Mimic user scroll
            await page.mouse.wheel(0, 200)
            await page.wait_for_timeout(1000)

            # Optional: Screenshot + HTML for debugging
            await page.screenshot(path=f"debug_{hash_key}.png", full_page=True)
            with open(f"debug_{hash_key}.html", "w", encoding="utf-8") as f:
                f.write(await page.content())

            structured_content = await extract_structured_content(page)
        finally:
            await browser.close()

    with open(cache_file, "wb") as f:
        pickle.dump(structured_content, f)
    with open(hash_file, "wb") as f:
        pickle.dump(hash_key, f)

    save_raw_content(link, structured_content)
    return structured_content


async def scrape_web_data(links):
    print(f"\n🌐 Starting web scraping for {len(links)} link(s)...")

    final_outputs = []

    for link in links:
        try:
            content = await fetch_or_cache_data(link)
            print(f"🤖 Running Gemini analysis for: {link}")

            table_prompt = create_table_prompt(content)
            faq_prompt = create_faq_prompt(content)

            table_response = model.generate_content(table_prompt)
            faq_response = model.generate_content(faq_prompt)

            final_output = (
                f"\n--- 🔍 Analysis for: {link} ---\n"
                f"\n📘 Table Breakdown:\n{table_response.text if table_response else '❌ Table breakdown failed.'}\n"
                f"\n❓ FAQs:\n{faq_response.text if faq_response else '❌ FAQ extraction failed.'}\n"
                f"\n--- ✅ END OF {link} ---\n"
            )

            final_outputs.append(final_output)

        except Exception as e:
            print(f"❌ Failed to process {link}: {str(e)}")
            final_outputs.append(f"\n--- ❌ Error processing {link} ---\nError: {str(e)}\n")

    return "\n".join(final_outputs)
