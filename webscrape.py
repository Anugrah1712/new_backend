import os
import asyncio
import pickle
import hashlib
import shutil
from crawl4ai import AsyncWebCrawler
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")

# Gemini model setup
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

# Caching paths
WEB_SCRAPE_PICKLE = "scraped_data.pkl"
LINKS_HASH_FILE = "links_hash.pkl"


def save_structured_content_to_file(link, content):
    os.makedirs("raw_structured_dumps", exist_ok=True)
    filename_hash = hashlib.md5(link.encode()).hexdigest()
    file_path = os.path.join("raw_structured_dumps", f"structured_{filename_hash}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Saved raw structured content to {file_path}")


def create_table_prompt(structured_content):
    return (
        "You are analyzing a web page with one or more interest rate tables related to Fixed Deposits (FDs). "
        "For EACH table in the content below:\n"
        "- Mention the table's heading/title or any label that identifies the table (e.g. 'FD MAX', 'Senior Citizens FD')\n"
        "- Interpret all rows and columns precisely.\n"
        "- Clearly explain what each column means. For instance:\n"
        "    * 'At maturity (p.a.)' → Interest rate applicable at maturity\n"
        "    * 'Monthly (p.a.)' → Effective annual interest rate if payout is monthly\n"
        "- For each row, summarize the interest rate for each payout option with a concrete sentence.\n"
        "    Example: 'For 12–14 months tenure, monthly payout gives 7.35% per annum.'\n"
        "- Highlight the highest available rate in the table and the corresponding tenure/payout.\n"
        "- Do NOT compare across tables. Each table should be explained independently.\n"
        "- If applicable, explain eligibility criteria mentioned above or near the table.\n\n"
        "Here is the content:\n\n" + structured_content
    )


def create_faq_prompt(structured_content):
    return (
        "From the content below, extract up to 20 Frequently Asked Questions (FAQs). "
        "Include both questions found in the content and logical questions a user might ask. Format:\n"
        "Q: <question>\nA: <answer>\n\n"
        "Content:\n\n" + structured_content
    )


semaphore = asyncio.Semaphore(1)  # Limit concurrency

async def process_link(crawler, link, use_markdown):
    async with semaphore:
        try:
            print(f"[INFO] Crawling: {link}")
            result = await crawler.arun(url=link)
            structured_content = result.markdown if use_markdown else result.html
            structured_content = structured_content or "No content extracted."

            save_structured_content_to_file(link, structured_content)

            table_prompt = create_table_prompt(structured_content)
            table_response = model.generate_content(table_prompt)
            table_details = table_response.text if table_response else "❌ Table breakdown failed."

            faq_prompt = create_faq_prompt(structured_content)
            faq_response = model.generate_content(faq_prompt)
            faq_text = faq_response.text if faq_response else "❌ FAQ extraction failed."

            return (
                f"\n\n--- Scraped Content from: {link} ---\n"
                f"\n📑 Raw Content Preview (first 1000 chars):\n{structured_content[:1000]}...\n"
                f"\n📘 Detailed Table Breakdown:\n{table_details}\n"
                f"\n❓ FAQs:\n{faq_text}\n"
                f"\n--- END OF PAGE ---\n"
            )

        except Exception as e:
            print(f"[ERROR] Failed to process {link}: {e}")
            return f"\n\n--- Scraped Content from: {link} ---\n❌ Error: {e}\n"


async def scrape_web_data(links=None, use_markdown=True):
    if not links and os.path.exists(WEB_SCRAPE_PICKLE):
        with open(WEB_SCRAPE_PICKLE, "rb") as f:
            cached_text = pickle.load(f)
            print("✅ Loaded cached data!")
            return cached_text

    new_links_str = ",".join(links) if links else ""
    new_hash = hashlib.md5(new_links_str.encode()).hexdigest()

    old_hash = None
    if os.path.exists(LINKS_HASH_FILE):
        with open(LINKS_HASH_FILE, "rb") as f:
            old_hash = pickle.load(f)

    # Always clear Playwright state before scraping
    print("[INFO] Cleaning up old Playwright cache...")
    shutil.rmtree("/root/.cache/ms-playwright", ignore_errors=True)

    # If hash differs, clear old cache
    if new_hash != old_hash:
        print("[INFO] Links changed. Deleting previous cache...")
        if os.path.exists(WEB_SCRAPE_PICKLE):
            os.remove(WEB_SCRAPE_PICKLE)
        if os.path.exists(LINKS_HASH_FILE):
            os.remove(LINKS_HASH_FILE)

    print("[INFO] Starting web scraping...")

    async with AsyncWebCrawler() as crawler:
        results = []
        for link in links:
            result = await process_link(crawler, link, use_markdown)
            results.append(result)

    scraped_text = "\n".join(results)

    with open(WEB_SCRAPE_PICKLE, "wb") as f:
        pickle.dump(scraped_text, f)
        print("💾 Saved new scraped data to cache.")

    with open(LINKS_HASH_FILE, "wb") as f:
        pickle.dump(new_hash, f)

    return scraped_text


# Example runner
async def main():
    final_data = await scrape_web_data()
    print("\n✅ Final Output Preview:\n", final_data[:2000])
    print("\n ✅ <----------------------------------------------------Final output preview ends here------------------------------------------------------------------>✅ ")

if __name__ == "__main__":
    asyncio.run(main())
