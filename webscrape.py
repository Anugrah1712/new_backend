import hashlib
import os
import pickle
import asyncio
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
import google.generativeai as genai
from dotenv import load_dotenv

from sitemap import discover_all_urls  # sitemap crawler from earlier

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

REQUEST_TIMEOUT = 20  # seconds

# Which browser TLS/HTTP fingerprint curl_cffi should impersonate. Cloudflare
# and similar WAFs fingerprint the TLS handshake itself (JA3), not just the
# User-Agent header — plain `requests`/urllib3 has a fingerprint that gets
# flagged even with a browser User-Agent set. curl_cffi reproduces a real
# Chrome handshake, which is why this clears the 403 that `requests` hit.
IMPERSONATE_PROFILE = os.getenv("SCRAPE_IMPERSONATE", "chrome119")

# Fallback budget used only when a site has no discoverable sitemap at all
# (in which case we fall back to pure link-crawling from the homepage).
MAX_PAGES = int(os.getenv("SCRAPE_MAX_PAGES", "60"))

# Hard ceiling on how many pages we'll ever scrape even when the sitemap
# declares far more (e.g. au.bank.in declares 3,349 URLs). Without this,
# a sitemap-seeded crawl of a large site would try to hit every single
# page, which is slow and burns a lot of Gemini calls for table summaries.
MAX_PAGES_CEILING = int(os.getenv("SCRAPE_MAX_PAGES_CEILING", "300"))

# URL substrings that mark "lower priority" content — pages containing any
# of these get pushed to the end of the queue and are the first to be cut
# when the sitemap total exceeds MAX_PAGES_CEILING. Extend as needed
# (e.g. "/press-release/", "/careers/").
LOW_PRIORITY_PATTERNS = ["/blog/"]

MAX_CONCURRENT_REQUESTS = int(os.getenv("SCRAPE_CONCURRENCY", "5"))


def hash_url(url):
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def normalize_url(url):
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
    """Blocking HTTP GET via curl_cffi, which impersonates a real browser's
    TLS fingerprint (not just headers) — needed because plain requests/urllib3
    gets blocked by Cloudflare/Akamai-style bot management even with a
    spoofed User-Agent. Always called via asyncio.to_thread so the rest of
    the file can stay async."""
    response = curl_requests.get(
        url,
        impersonate=IMPERSONATE_PROFILE,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type:
        raise ValueError(f"Skipping non-HTML content ({content_type}) at {url}")
    return response.text


def _prioritize_urls(urls):
    """Sorts URLs so 'core' pages come before low-priority ones (blog posts
    etc). Used to decide what survives when MAX_PAGES_CEILING forces a cut —
    we want the homepage/products/FAQs kept over a random slice of 2,000
    blog posts, not whatever order a Python set happens to produce."""
    def is_low_priority(url):
        return any(pattern in url for pattern in LOW_PRIORITY_PATTERNS)

    return sorted(urls, key=is_low_priority)  # False (0) sorts before True (1)


def get_effective_crawl_plan(main_url):
    """Discovers sitemap URLs and returns (seed_urls, effective_max_pages).
    seed_urls is None if no sitemap was found, signalling the caller to fall
    back to homepage-only link-crawling with the flat MAX_PAGES budget."""
    sitemap_urls = discover_all_urls(main_url, verbose=False)

    if not sitemap_urls:
        print(f"[SITEMAP] None found for {main_url} — falling back to "
              f"link-crawl from homepage with MAX_PAGES={MAX_PAGES}.")
        return None, MAX_PAGES

    ordered = _prioritize_urls(sitemap_urls)
    capped = len(ordered) > MAX_PAGES_CEILING
    effective = min(len(ordered), MAX_PAGES_CEILING)

    print(f"[SITEMAP] {len(ordered)} URLs declared for {main_url}"
          + (f", capped to {effective} (low-priority pages dropped first)" if capped else "")
          + ".")

    return ordered[:effective], effective


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

    seed_urls, effective_max_pages = get_effective_crawl_plan(main_url)

    if seed_urls:
        # Sitemap-seeded: start the queue with prioritized, capped sitemap
        # URLs rather than relying purely on link-following from the
        # homepage, so pages with no internal inbound links still get hit.
        to_scrape = {normalize_url(u) for u in seed_urls}
    else:
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

    while to_scrape and len(scraped) < effective_max_pages:
        # Pull a batch (bounded by remaining page budget) and fetch it concurrently
        # instead of one page at a time — this is what was making a full crawl take hours.
        remaining_budget = effective_max_pages - len(scraped)
        batch = [to_scrape.pop() for _ in range(min(len(to_scrape), remaining_budget))]

        print(f"[BATCH] Fetching {len(batch)} pages (scraped so far: {len(scraped)}/{effective_max_pages})")
        batch_results = await asyncio.gather(*(process(url) for url in batch))

        for url, result, links in batch_results:
            scraped.add(url)
            if result:
                results.append(result)
            for link in links:
                norm_link = normalize_url(link)
                if norm_link not in scraped and norm_link not in to_scrape:
                    to_scrape.add(norm_link)

    if len(scraped) >= effective_max_pages and to_scrape:
        print(f"[CRAWL] Hit effective_max_pages={effective_max_pages} cap — {len(to_scrape)} more discovered URLs were left unvisited. "
              f"Raise SCRAPE_MAX_PAGES_CEILING if you need a deeper crawl.")

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