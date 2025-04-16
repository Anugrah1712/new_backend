import os
import asyncio
import pickle
import hashlib
from dotenv import load_dotenv
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import google.generativeai as genai

# Load environment variables
print("[DEBUG] Loading environment variables...")
load_dotenv()
genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")
print("[DEBUG] Gemini configured with API key.")

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

WEB_SCRAPE_PICKLE = "scraped_data.pkl"
LINKS_HASH_FILE = "links_hash.pkl"

def save_structured_content_to_file(link, content):
    print(f"[DEBUG] Saving structured content for {link}...")
    os.makedirs("raw_structured_dumps", exist_ok=True)
    filename_hash = hashlib.md5(link.encode()).hexdigest()
    with open(f"raw_structured_dumps/structured_{filename_hash}.html", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Saved structured content for {link}")

def create_table_prompt(structured_content):
    print("[DEBUG] Creating table analysis prompt...")
    return (
        "You are analyzing a web page with one or more interest rate tables related to Fixed Deposits (FDs). "
        "For EACH table in the content below:\n"
        "- Mention the table's heading/title or any label that identifies the table (e.g. 'FD MAX', 'Senior Citizens FD')\n"
        "- Interpret all rows and columns precisely.\n"
        "- Clearly explain what each column means.\n"
        "- For each row, summarize the interest rate for each payout option with a concrete sentence.\n"
        "- Highlight the highest available rate and its tenure/payout.\n"
        "- Do NOT compare across tables.\n"
        "- If applicable, explain eligibility criteria mentioned above or near the table.\n\n"
        "Here is the content:\n\n" + structured_content
    )

def create_faq_prompt(structured_content):
    print("[DEBUG] Creating FAQ extraction prompt...")
    return (
        "From the content below, extract up to 20 Frequently Asked Questions (FAQs). "
        "Include both questions found in the content and logical questions a user might ask. Format:\n"
        "Q: <question>\nA: <answer>\n\n"
        "Content:\n\n" + structured_content
    )

async def playwright_scrape(url: str) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            await page.goto(url, timeout=30000)
            content = await page.content()
            await browser.close()
            return content
    except PlaywrightTimeoutError:
        raise Exception("Timeout while loading the page.")
    except Exception as e:
        raise e

async def scrape_single_link(link, retries=3):
    attempt = 0
    while attempt < retries:
        try:
            print(f"[INFO] Attempting to scrape: {link}, Attempt {attempt + 1}")
            content = await playwright_scrape(link)
            save_structured_content_to_file(link, content)

            table_response = model.generate_content(create_table_prompt(content)).text
            faq_response = model.generate_content(create_faq_prompt(content)).text

            print(f"[INFO] Successfully scraped: {link}")
            return (
                f"\n\n--- Scraped Content from: {link} ---\n"
                f"\n📁 Raw Content Preview (first 1000 chars):\n{content[:1000]}...\n"
                f"\n📘 Detailed Table Breakdown:\n{table_response}\n"
                f"\n❓ FAQs:\n{faq_response}\n"
                f"\n--- END OF PAGE ---\n"
            )

        except Exception as e:
            print(f"[ERROR] Failed to process {link} on attempt {attempt + 1}: {e}")
            attempt += 1
            if attempt < retries:
                print("[INFO] Retrying...")
                await asyncio.sleep(2)
            else:
                print(f"[ERROR] Failed after {retries} attempts.")
                return f"\n\n--- Scraped Content from: {link} ---\n❌ Error: {e}\n"

async def scrape_web_data(links=None):
    if not links and os.path.exists(WEB_SCRAPE_PICKLE):
        with open(WEB_SCRAPE_PICKLE, "rb") as f:
            print("✅ Loaded cached scraped data.")
            return pickle.load(f)

    new_links_str = ",".join(links or [])
    new_hash = hashlib.md5(new_links_str.encode()).hexdigest()
    print(f"[DEBUG] Generated hash for links: {new_hash}")

    if os.path.exists(LINKS_HASH_FILE):
        with open(LINKS_HASH_FILE, "rb") as f:
            old_hash = pickle.load(f)
        if new_hash == old_hash and os.path.exists(WEB_SCRAPE_PICKLE):
            print("✅ No link change. Loading from cache.")
            with open(WEB_SCRAPE_PICKLE, "rb") as f:
                return pickle.load(f)

    print("[INFO] Starting fresh scraping...")
    results = [await scrape_single_link(link) for link in links]

    combined_text = "\n".join(results)
    print("[DEBUG] Writing scraped results to cache files...")
    with open(WEB_SCRAPE_PICKLE, "wb") as f:
        pickle.dump(combined_text, f)
    with open(LINKS_HASH_FILE, "wb") as f:
        pickle.dump(new_hash, f)

    print("💾 Scraping done. Data cached.")
    return combined_text

if __name__ == "__main__":
    print("[DEBUG] Starting script execution with example links...")
    asyncio.run(scrape_web_data())
