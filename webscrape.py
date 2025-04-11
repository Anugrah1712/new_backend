import os
import asyncio
import pickle
import hashlib
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

# # Save raw markdown/html dump
# def save_structured_content_to_file(link, content):
#     os.makedirs("raw_structured_dumps", exist_ok=True)
#     filename_hash = hashlib.md5(link.encode()).hexdigest()
#     file_path = os.path.join("raw_structured_dumps", f"structured_{filename_hash}.txt")
#     with open(file_path, "w", encoding="utf-8") as f:
#         f.write(content)
#     print(f"📄 Saved raw structured content to {file_path}")

# Scraper function
async def scrape_web_data(links=None, use_markdown=True):
    if not links and os.path.exists(WEB_SCRAPE_PICKLE):
        with open(WEB_SCRAPE_PICKLE, "rb") as f:
            cached_text = pickle.load(f)
            print("✅ Loaded cached data!")
            return cached_text

    new_links_str = ",".join(links) if links else ""
    new_hash = hashlib.md5(new_links_str.encode()).hexdigest()

    if os.path.exists(LINKS_HASH_FILE):
        with open(LINKS_HASH_FILE, "rb") as f:
            old_hash = pickle.load(f)
        if new_hash == old_hash and os.path.exists(WEB_SCRAPE_PICKLE):
            with open(WEB_SCRAPE_PICKLE, "rb") as f:
                cached_text = pickle.load(f)
                print("✅ No link change. Loaded cached data!")
                return cached_text

    print("[INFO] Starting web scraping...")
    scraped_text = ""

    async with AsyncWebCrawler() as crawler:
        for link in links:
            try:
                print(f"[INFO] Crawling: {link}")
                result = await crawler.arun(url=link)
                structured_content = result.markdown if use_markdown else result.html
                structured_content = structured_content or "No content extracted."

                # save_structured_content_to_file(link, structured_content)

                # 🔍 Prompt 1: Exhaustive table-wise breakdown
                table_prompt = (
                    "You are analyzing a web page with interest rate tables. "
                    "For EACH table in the content below, write a full breakdown. For each table:\n"
                    "- Mention the heading/title\n"
                    "- Explain each column (tenure, rate, payout frequency, etc.)\n"
                    "- Describe values (e.g. '6.75% interest for 18 months FD with monthly payout')\n"
                    "- Call out special cases like highest rate, eligibility criteria, etc.\n"
                    "- DO NOT compare tables; treat them as separate blocks.\n\n"
                    "Below is the page content:\n\n"
                    + structured_content
                )
                table_response = model.generate_content(table_prompt)
                table_details = table_response.text if table_response else "❌ Table breakdown failed."

                # 🔍 Prompt 2: Extract up to 20 FAQs
                faq_prompt = (
                    "From the content below, extract up to 20 Frequently Asked Questions (FAQs). "
                    "Include both questions found in the content and logical questions a user might ask. Format:\n"
                    "Q: <question>\nA: <answer>\n\n"
                    "Content:\n\n" + structured_content
                )
                faq_response = model.generate_content(faq_prompt)
                faq_text = faq_response.text if faq_response else "❌ FAQ extraction failed."

                # 🔗 Combine results
                scraped_text += (
                    f"\n\n--- Scraped Content from: {link} ---\n"
                    f"\n📑 Raw Content Preview (first 1000 chars):\n{structured_content[:1000]}...\n"
                    f"\n📘 Detailed Table Breakdown:\n{table_details}\n"
                    f"\n❓ FAQs:\n{faq_text}\n"
                    f"\n--- END OF PAGE ---\n"
                )

            except Exception as e:
                print(f"[ERROR] Failed to process {link}: {e}")
    # Cache result
    with open(WEB_SCRAPE_PICKLE, "wb") as f:
        pickle.dump(scraped_text, f)
        print("💾 Saved new scraped data to cache.")

    with open(LINKS_HASH_FILE, "wb") as f:
        pickle.dump(new_hash, f)

    return scraped_text

# Runner
async def main():
    final_data = await scrape_web_data()
    print("\n✅ Final Output Preview:\n", final_data[:2000])  # Show first 2K chars
    print("\n ✅ <----------------------------------------------------Final output preview ends here------------------------------------------------------------------>✅ ")

if __name__ == "__main__":
    asyncio.run(main())
