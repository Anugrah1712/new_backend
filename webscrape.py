import os
import asyncio
import pickle
import hashlib
from dotenv import load_dotenv
from crawl4ai import SmartScraper  # Requires crawl4ai >= 0.1.26
import google.generativeai as genai

print("[DEBUG] Loading environment variables...")
load_dotenv()
print("[DEBUG] Environment loaded.")

print("[DEBUG] Configuring Gemini...")
genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")
print("[DEBUG] Gemini configured successfully.")

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

scraper = SmartScraper()

def create_table_prompt(structured_content):
    print("[DEBUG] Creating table prompt for Gemini...")
    return (
        "You are analyzing a web page with one or more interest rate tables related to Fixed Deposits (FDs). "
        "For EACH table in the content below:\n"
        "- Mention the table's heading/title or any label that identifies the table\n"
        "- Interpret all rows and columns precisely.\n"
        "- Clearly explain what each column means.\n"
        "- For each row, summarize the interest rate.\n"
        "- Highlight the highest available rate and its tenure/payout.\n"
        "- Do NOT compare across tables.\n"
        "- If applicable, explain eligibility criteria mentioned above or near the table.\n\n"
        "Here is the content:\n\n" + structured_content
    )

def create_faq_prompt(structured_content):
    print("[DEBUG] Creating FAQ prompt for Gemini...")
    return (
        "From the content below, extract up to 20 Frequently Asked Questions (FAQs). "
        "Include both visible questions and logical ones a user might ask.\n"
        "Q: <question>\nA: <answer>\n\n"
        "Content:\n\n" + structured_content
    )

async def scrape_single_link(link):
    try:
        print(f"[INFO] Scraping URL: {link}")
        content = await scraper.run(link)
        html_content = content["output"]
        print(f"[DEBUG] Successfully scraped structured HTML from {link}. Length: {len(html_content)} characters")

        filename_hash = hashlib.md5(link.encode()).hexdigest()
        os.makedirs("raw_structured_dumps", exist_ok=True)
        file_path = f"raw_structured_dumps/structured_{filename_hash}.html"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"[DEBUG] Saved structured HTML to {file_path}")

        print("[DEBUG] Sending table content to Gemini...")
        table_response = model.generate_content(create_table_prompt(html_content)).text
        print("[DEBUG] Received table breakdown from Gemini.")

        print("[DEBUG] Sending FAQ content to Gemini...")
        faq_response = model.generate_content(create_faq_prompt(html_content)).text
        print("[DEBUG] Received FAQs from Gemini.")

        return (
            f"\n\n--- Scraped Content from: {link} ---\n"
            f"\n📁 Raw Content Preview (first 1000 chars):\n{html_content[:1000]}...\n"
            f"\n📘 Table Breakdown:\n{table_response}\n"
            f"\n❓ FAQs:\n{faq_response}\n"
            f"\n--- END OF PAGE ---\n"
        )
    except Exception as e:
        print(f"[ERROR] Failed to scrape {link}: {e}")
        return f"\n\n--- Scraped Content from: {link} ---\n❌ Error: {e}\n"

async def scrape_web_data(links):
    print(f"[DEBUG] Received links to scrape: {links}")
    new_links_str = ",".join(links)
    new_hash = hashlib.md5(new_links_str.encode()).hexdigest()
    print(f"[DEBUG] Computed hash for current links: {new_hash}")

    if os.path.exists(LINKS_HASH_FILE):
        print("[DEBUG] Checking existing hash...")
        with open(LINKS_HASH_FILE, "rb") as f:
            old_hash = pickle.load(f)
        if new_hash == old_hash and os.path.exists(WEB_SCRAPE_PICKLE):
            print("✅ No link changes detected. Loading cached result...")
            with open(WEB_SCRAPE_PICKLE, "rb") as f:
                return pickle.load(f)
        else:
            print("[DEBUG] Link hash changed. Proceeding with fresh scrape.")
    else:
        print("[DEBUG] No previous hash file found. Proceeding with scrape.")

    print("[INFO] Starting scraping for all links...")
    results = await asyncio.gather(*[scrape_single_link(link) for link in links])
    combined_text = "\n".join(results)

    print("[DEBUG] Saving scraped content and hash to cache files...")
    with open(WEB_SCRAPE_PICKLE, "wb") as f:
        pickle.dump(combined_text, f)
    with open(LINKS_HASH_FILE, "wb") as f:
        pickle.dump(new_hash, f)

    print("✅ Scraping complete and data cached.")
    return combined_text

if __name__ == "__main__":
    print("[DEBUG] Starting scraping execution...")
   
    
    asyncio.run(scrape_web_data())
