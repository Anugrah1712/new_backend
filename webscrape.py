import hashlib
import os
import pickle
import asyncio
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Cache file
CACHE_PATH = "scraped_cache.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        scraped_cache = pickle.load(f)
else:
    scraped_cache = {}

# Plain requests.get() with no headers gets blocked/served different content by
# many sites (bot detection, CDNs). A normal browser User-Agent avoids most of that.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 20  # seconds

# Hard cap on how many pages a single crawl will ever visit. Without this,
# a site with tracking/query params, calendars, or locator pages
# (?state=X&city=Y for every city) can generate effectively unlimited
# "new" URLs and the crawl never finishes. Override with an env var if needed.
MAX_PAGES = int(os.getenv("SCRAPE_MAX_PAGES", "60"))

# How many pages to fetch at the same time. Was 1-at-a-time before, which is
# why a large site took hours — most of that time was just waiting on network
# round trips serially instead of in parallel.
MAX_CONCURRENT_REQUESTS = int(os.getenv("SCRAPE_CONCURRENCY", "5"))


def hash_url(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def normalize_url(url):
    """Collapse https://site.com/page, https://site.com/page/, and
    https://site.com/page?ref=abc&utm_source=x into one URL. Without this,
    query-string/tracking params make the crawler treat the same page as
    endless "new" pages — the most common reason a crawl never terminates."""
    try:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        normalized = parsed._replace(path=path, query="", fragment="")
        return normalized.geturl()
    except:
        return url


def is_internal_link(base_url, link):
    try:
        parsed_base = urlparse(base_url)
        parsed_link = urlparse(link)
        return (parsed_base.netloc == parsed_link.netloc) or (parsed_link.netloc == "")
    except:
        return False


def _fetch_html(url):
    """Blocking HTTP GET. Always called via asyncio.to_thread so the rest of the
    file can stay async without needing an async HTTP client."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        raise ValueError(f"Skipping non-HTML content ({content_type}) at {url}")
    return response.text


async def scrape_page(url):
    url_hash = hash_url(url)
    if url_hash in scraped_cache:
        print(f"[CACHE] Using cached data for {url}")
        return scraped_cache[url_hash]

    print(f"[SCRAPE] Scraping {url}")
    try:
        html = await asyncio.to_thread(_fetch_html, url)
        soup = BeautifulSoup(html, "lxml")

        # Extract tables
        tables = soup.find_all("table")
        table_data_list = []
        for i, table in enumerate(tables, start=1):
            rows = table.find_all("tr")
            table_data = []
            for row in rows:
                cols = row.find_all("td")
                col_text = [col.get_text(strip=True) for col in cols if col]
                if col_text:
                    table_data.append(col_text)
            if table_data:
                summary = await convert_table_to_sentences_gemini(table_data, i)
                table_data_list.append(summary)

        # Extract full text
        body = soup.find("body")
        full_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

        # Extract FAQs
        # NOTE: unlike Playwright, BeautifulSoup only sees the static HTML that
        # was returned by the server — there's no "click to expand" step because
        # there's no live page to click on. If the FAQ answers are only inserted
        # into the DOM by JavaScript on click, they won't be present here and
        # this block will simply find nothing. If the site renders the Q&A HTML
        # up front and just hides it with CSS (common for AEM-style accordions),
        # this will still pick it up.
        faqs = []
        try:
            faq_container = soup.select_one(".faqs.aem-GridColumn.aem-GridColumn--default--12")
            if faq_container:
                questions = faq_container.select(".accordion_toggle, .accordion_row")
                answers = faq_container.select(".accordion_body, .accordionbody_links, .aem-rte-content")
                for question_el, answer_el in zip(questions, answers):
                    question = question_el.get_text(strip=True)
                    answer = answer_el.get_text(strip=True)
                    if question and answer:
                        faqs.append({"question": question, "answer": answer})
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
        print("Scraped Data--------------------------->", result)
        return result

    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return None


async def get_internal_links(url):
    try:
        html = await asyncio.to_thread(_fetch_html, url)
    except Exception as e:
        print(f"[LINKS] Failed to fetch {url} for link extraction: {e}")
        return []

    soup = BeautifulSoup(html, "lxml")
    links = set()
    for anchor in soup.find_all("a"):
        try:
            href = anchor.get("href")
            if href and is_internal_link(url, href):
                absolute_url = urljoin(url, href.split("#")[0])
                links.add(absolute_url)
        except:
            continue

    print(f"[INTERNAL LINKS] Found {len(links)} links in {url}")
    return list(links)


async def scrape_with_playwright_recursive(main_url):
    # Function name kept as-is for compatibility with existing callers
    # (scrape_web_data below, and anything else importing it directly),
    # even though it no longer launches Playwright/a browser.
    main_url = normalize_url(main_url)
    to_scrape = {main_url}
    scraped = set()
    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def process(url):
        async with semaphore:
            print(f"[VISIT] {url}")
            result = await scrape_page(url)
            links = []
            if result:
                try:
                    links = await get_internal_links(url)
                except Exception as e:
                    print(f"[LINKS] Failed to extract internal links from {url}: {e}")
            return url, result, links

    while to_scrape and len(scraped) < MAX_PAGES:
        # Pull a batch (bounded by remaining page budget) and fetch it concurrently
        # instead of one page at a time — this is what was making a full crawl take hours.
        remaining_budget = MAX_PAGES - len(scraped)
        batch = [to_scrape.pop() for _ in range(min(len(to_scrape), remaining_budget))]

        print(f"[BATCH] Fetching {len(batch)} pages (scraped so far: {len(scraped)}/{MAX_PAGES})")
        batch_results = await asyncio.gather(*(process(url) for url in batch))

        for url, result, links in batch_results:
            scraped.add(url)
            if result:
                results.append(result)
            for link in links:
                norm_link = normalize_url(link)
                if norm_link not in scraped and norm_link not in to_scrape:
                    to_scrape.add(norm_link)

    if len(scraped) >= MAX_PAGES and to_scrape:
        print(f"[CRAWL] Hit MAX_PAGES={MAX_PAGES} cap — {len(to_scrape)} more discovered URLs were left unvisited. "
              f"Raise SCRAPE_MAX_PAGES if you need a deeper crawl.")

    _print_crawl_summary(main_url, results)
    return results


def _print_crawl_summary(main_url, results):
    """Prints one consolidated summary of everything scraped in this run,
    so it's easy to see at a glance whether the crawl actually pulled
    real content or just bot-check / empty pages."""
    total_pages = len(results)
    total_words = sum(len(r.get("full_text", "").split()) for r in results)
    total_tables = sum(len(r.get("table_summaries", [])) for r in results)
    total_faqs = sum(len(r.get("faqs", [])) for r in results)

    print("\n" + "=" * 70)
    print(f"[CRAWL SUMMARY] Root URL: {main_url}")
    print(f"Pages scraped   : {total_pages}")
    print(f"Total words     : {total_words}")
    print(f"Total tables    : {total_tables}")
    print(f"Total FAQs      : {total_faqs}")
    print("-" * 70)
    if not results:
        print("⚠️  No pages were scraped — check the URL and any bot-blocking (e.g. Cloudflare).")
    for i, r in enumerate(results, start=1):
        word_count = len(r.get("full_text", "").split())
        table_count = len(r.get("table_summaries", []))
        faq_count = len(r.get("faqs", []))
        flag = " ⚠️ low content" if word_count < 50 else ""
        print(f"{i:>3}. {r.get('url')}")
        print(f"     words={word_count}  tables={table_count}  faqs={faq_count}{flag}")
    print("=" * 70 + "\n")


async def convert_table_to_sentences_gemini(table_data, index):
    table_input = f"Table {index}:\n" + "\n".join([", ".join(row) for row in table_data])
    chat = model.start_chat(history=[
        {"role": "user", "parts": [
            "Convert table to descriptive sentences.\n"

        ]},
        {"role": "model", "parts": ["Please provide the table."]}
    ])
    # chat.send_message is a synchronous call under the hood — run it in a
    # thread so it doesn't block the event loop the way `await` on it did before.
    response = await asyncio.to_thread(chat.send_message, table_input)
    return response.text


async def scrape_web_data(url):
    return await scrape_with_playwright_recursive(url)