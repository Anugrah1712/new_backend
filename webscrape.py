import os
import asyncio
import pickle
import hashlib
from dotenv import load_dotenv
from playwright.async_api import async_playwright
import google.generativeai as genai

# Load .env and Gemini API key
load_dotenv()
genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")

# Gemini setup
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

# Cache files
WEB_SCRAPE_PICKLE = "scraped_data.pkl"
LINKS_HASH_FILE = "links_hash.pkl"

# Save content
def save_structured_content_to_file(link, content):
    os.makedirs("raw_structured_dumps", exist_ok=True)
    filename_hash = hashlib.md5(link.encode()).hexdigest()
    with open(f"raw_structured_dumps/structured_{filename_hash}.html", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Saved structured content for {link}")

# Gemini prompts
def create_table_prompt(content):
    return (
        "You are analyzing a web page with one or more interest rate tables related to Fixed Deposits (FDs). "
        "For EACH table in the content below:\n"
        "- Mention title\n- Explain all columns and rows\n"
        "- Summarize payouts\n- Highlight highest rate\n- DO NOT compare tables\n\n"
        "Here is the content:\n\n" + content
    )

def create_faq_prompt(content):
    return (
        "From the content below, extract up to 20 FAQs. Include logical and visible questions.\n"
        "Format:\nQ: <question>\nA: <answer>\n\nContent:\n\n" + content
    )

# Scrape one link using Playwright
async def scrape_single_link(link):
    try:
        print(f"[INFO] Scraping: {link}")
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                           "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                locale="en-US",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.google.com/"
                }
            )
            page = await context.new_page()
            await page.goto(link, timeout=60000)
            await page.wait_for_timeout(3000)  # Let dynamic content load
            content = await page.content()
            await browser.close()

        save_structured_content_to_file(link, content)

        # Gemini Prompts
        table_prompt = create_table_prompt(content)
        table_response = model.generate_content(table_prompt).text

        faq_prompt = create_faq_prompt(content)
        faq_response = model.generate_content(faq_prompt).text

        return (
            f"\n\n--- Scraped Content from: {link} ---\n"
            f"\n📑 Raw Content Preview (first 1000 chars):\n{content[:1000]}...\n"
            f"\n📘 Detailed Table Breakdown:\n{table_response}\n"
            f"\n❓ FAQs:\n{faq_response}\n"
            f"\n--- END OF PAGE ---\n"
        )

    except Exception as e:
        print(f"[ERROR] Failed to scrape {link}: {e}")
        return f"\n\n--- Scraped Content from: {link} ---\n❌ Error: {e}\n"


# Main scrape function
async def scrape_web_data(links=None):
    if not links and os.path.exists(WEB_SCRAPE_PICKLE):
        with open(WEB_SCRAPE_PICKLE, "rb") as f:
            print("✅ Loaded cached scraped data.")
            return pickle.load(f)

    new_links_str = ",".join(links or [])
    new_hash = hashlib.md5(new_links_str.encode()).hexdigest()

    if os.path.exists(LINKS_HASH_FILE):
        with open(LINKS_HASH_FILE, "rb") as f:
            old_hash = pickle.load(f)
        if new_hash == old_hash and os.path.exists(WEB_SCRAPE_PICKLE):
            with open(WEB_SCRAPE_PICKLE, "rb") as f:
                print("✅ No link change. Loaded cached data.")
                return pickle.load(f)

    print("[INFO] Starting fresh web scraping...")
    results = [await scrape_single_link(link) for link in links]

    combined_text = "\n".join(results)
    with open(WEB_SCRAPE_PICKLE, "wb") as f:
        pickle.dump(combined_text, f)
    with open(LINKS_HASH_FILE, "wb") as f:
        pickle.dump(new_hash, f)

    print("💾 Scraping done. Data cached.")
    return combined_text

# For direct execution
if __name__ == "__main__":
    asyncio.run(scrape_web_data())
#######################################done######################