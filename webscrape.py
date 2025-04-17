import os
import asyncio
import pickle
import hashlib
from dotenv import load_dotenv
import google.generativeai as genai
from crawl4ai import AsyncWebCrawler

# Monkey-patch Playwright args for crawl4ai
original_init = AsyncWebCrawler.__init__

def patched_init(self, *args, **kwargs):
    if "playwright_browser_args" not in kwargs:
        kwargs["playwright_browser_args"] = ["--no-sandbox", "--disable-setuid-sandbox"]
    original_init(self, *args, **kwargs)

AsyncWebCrawler.__init__ = patched_init

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyBNJvzSaKq26JHLLMSlIYaZAzOANtc8FCY")  # replace with your actual key

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

SCRAPED_DIR = "scraped_links"
os.makedirs(SCRAPED_DIR, exist_ok=True)

def save_structured_content_to_file(link, content):
    os.makedirs("raw_structured_dumps", exist_ok=True)
    filename_hash = hashlib.md5(link.encode()).hexdigest()
    file_path = os.path.join("raw_structured_dumps", f"structured_{filename_hash}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Saved raw structured content to {file_path}")

def get_cache_path_for_link(link):
    filename_hash = hashlib.md5(link.encode()).hexdigest()
    return os.path.join(SCRAPED_DIR, f"{filename_hash}.pkl")

async def scrape_web_data(links=None, use_markdown=True):
    if not links:
        print("[WARN] No links provided.")
        return ""

    scraped_text = ""

    for link in links:
        cache_path = get_cache_path_for_link(link)

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached_content = pickle.load(f)
                scraped_text += cached_content
                print(f"✅ Cache hit: Skipping scrape for {link}")
                continue

        retries = 3
        for attempt in range(retries):
            try:
                print(f"[INFO] Crawling: {link}")
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=link)

                structured_content = result.markdown if use_markdown else result.html
                structured_content = structured_content or "No content extracted."

                save_structured_content_to_file(link, structured_content)

                # Table Breakdown Prompt
                table_prompt = (
                    "You are analyzing a web page with interest rate tables. "
                    "For EACH table in the content below, write a full breakdown. For each table:\n"
                    "- Mention the heading/title\n"
                    "- Explain each column (tenure, rate, payout frequency, etc.)\n"
                    "- Mention the rate of interest for each payout options\n"
                    "- Describe values (e.g. '6.75% interest for 18 months FD with monthly payout')\n"
                    "- Call out special cases like highest rate, eligibility criteria, etc.\n"
                    "- DO NOT compare tables; treat them as separate blocks.\n\n"
                    "Below is the page content:\n\n"
                    + structured_content
                )
                table_response = model.generate_content(table_prompt)
                table_details = table_response.text if table_response else "❌ Table breakdown failed."

                # FAQ Extraction Prompt
                faq_prompt = (
                    "From the content below, extract up to 15 Frequently Asked Questions (FAQs). "
                    "Include both questions found in the content and logical questions a user might ask. Format:\n"
                    "Q: <question>\nA: <answer>\n\n"
                    "Content:\n\n" + structured_content
                )
                faq_response = model.generate_content(faq_prompt)
                faq_text = faq_response.text if faq_response else "❌ FAQ extraction failed."

                # Combine output
                link_output = (
                    f"\n\n--- Scraped Content from: {link} ---\n"
                    f"\n📑 Raw Content Preview (first 2000 chars):\n{structured_content[:2000]}...\n"
                    f"\n📘 Detailed Table Breakdown:\n{table_details}\n"
                    f"\n❓ FAQs:\n{faq_text}\n"
                    f"\n--- END OF PAGE ---\n"
                )

                with open(cache_path, "wb") as f:
                    pickle.dump(link_output, f)
                    print(f"💾 Cached scraped content for {link}")

                scraped_text += link_output
                break  # success: exit retry loop

            except Exception as e:
                print(f"[ERROR] Failed to process {link} on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    print(f"[INFO] Retrying scrape for {link}...")
                else:
                    print(f"[ERROR] Failed to process {link} after {retries} attempts.")

    return scraped_text

# Optional test runner
async def main():
   
    final_output = await scrape_web_data()
    print(final_output[:3000])  # Preview

if __name__ == "__main__":
    asyncio.run(main())
