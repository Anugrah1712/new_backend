import hashlib
import os
import pickle
import re
import time
import asyncio
from collections import deque
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from google import genai
from dotenv import load_dotenv

from sitemap import discover_all_urls  # sitemap crawler from earlier

load_dotenv()
# Gemini setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
genai_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "5"))  # free tier = 5 req/min for gemini-2.5-flash
_gemini_min_interval = 60.0 / GEMINI_RPM
_gemini_lock = asyncio.Lock()
_gemini_last_call = 0.0

# Circuit breaker. The free tier has TWO separate 429s: a per-minute rate limit
# (retryable — the API even tells you how long to wait) and a per-DAY project
# quota (GenerateRequestsPerDayPerProjectPerModel-FreeTier, 20 req/day for
# gemini-2.5-flash). Retrying the daily one is pointless: it doesn't reset until
# midnight Pacific, so each retry just burns ~2 minutes of wall clock and fails.
# Once we see it, stop calling Gemini for the rest of the run and use the raw
# table fallback instead.
_gemini_disabled = False
_gemini_disabled_reason = ""

# Summaries keyed by a hash of the table's own contents, so the same fee/nav
# table repeated across 40 pages costs one call, not 40 — and a re-run costs 0.
TABLE_CACHE_PATH = "table_summary_cache.pkl"
if os.path.exists(TABLE_CACHE_PATH):
    with open(TABLE_CACHE_PATH, "rb") as f:
        table_summary_cache = pickle.load(f)
else:
    table_summary_cache = {}


# Cache file
CACHE_PATH = "scraped_cache.pkl"
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "rb") as f:
        scraped_cache = pickle.load(f)
else:
    scraped_cache = {}

# Pages scraped since the last pickle flush. The whole cache dict is rewritten
# on every flush, so flushing per page is O(n^2) writes over a crawl — batch it.
_cache_dirty = 0
CACHE_FLUSH_EVERY = int(os.getenv("SCRAPE_CACHE_FLUSH_EVERY", "10"))

REQUEST_TIMEOUT = 20  # seconds

# Which browser TLS/HTTP fingerprint curl_cffi should impersonate. Cloudflare
# and similar WAFs fingerprint the TLS handshake itself (JA3), not just the
# User-Agent header — plain `requests`/urllib3 has a fingerprint that gets
# flagged even with a browser User-Agent set. curl_cffi reproduces a real
# Chrome handshake, which is why this clears the 403 that `requests` hit.
IMPERSONATE_PROFILE = os.getenv("SCRAPE_IMPERSONATE", "chrome119")

# When a site serves an interactive Cloudflare challenge (cf-mitigated: challenge),
# no TLS/JA3 impersonation can get past it — curl_cffi never executes the JS
# challenge. SCRAPER_API_KEY switches fetches to a third-party scraping API
# (ScraperAPI here; swap SCRAPER_API_URL / param names for ZenRows/ScrapingBee/etc.
# if you use a different provider) that solves the challenge server-side and
# returns rendered HTML. If unset, we use the direct curl_cffi request.
SCRAPER_API_KEY = os.getenv("SCRAPER_API_KEY")
SCRAPER_API_URL = "https://api.scraperapi.com"
SCRAPER_API_RENDER = os.getenv("SCRAPER_API_RENDER", "true")
SCRAPER_API_TIMEOUT = int(os.getenv("SCRAPER_API_TIMEOUT", "90"))  # renders are slow

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
    except ValueError:
        return url


def is_internal_link(base_url, link):
    try:
        parsed_base = urlparse(base_url)
        parsed_link = urlparse(link)
    except ValueError:
        return False
    if parsed_link.scheme and parsed_link.scheme not in ("http", "https"):
        return False  # mailto:, tel:, javascript: ...
    return (parsed_base.netloc == parsed_link.netloc) or (parsed_link.netloc == "")


def _flush_cache(force=False):
    global _cache_dirty
    if _cache_dirty and (force or _cache_dirty >= CACHE_FLUSH_EVERY):
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(scraped_cache, f)
        _cache_dirty = 0


def _fetch_via_scraper_api(url):
    """Routes the fetch through the scraping API, which solves the JS challenge
    server-side. Returns rendered HTML."""
    params = {
        "api_key": SCRAPER_API_KEY,
        "url": url,
        "render": SCRAPER_API_RENDER,
    }
    response = curl_requests.get(
        SCRAPER_API_URL, params=params, timeout=SCRAPER_API_TIMEOUT
    )
    response.raise_for_status()
    return response.text


def _fetch_html(url):
    """Blocking HTTP GET. Prefers the scraping API when SCRAPER_API_KEY is set
    (it's the only thing that gets past an interactive Cloudflare challenge);
    otherwise goes direct via curl_cffi, impersonating a real browser's TLS
    fingerprint, with fallback profiles and optional proxy routing (see
    sitemap.py's _fetch for why — deployment IPs get blocked by Cloudflare
    independent of fingerprint). Always called via asyncio.to_thread."""
    from sitemap import IMPERSONATE_FALLBACKS, SCRAPE_PROXY  # reuse same config

    if SCRAPER_API_KEY:
        try:
            return _fetch_via_scraper_api(url)
        except Exception as e:
            print(f"[FETCH] Scraping API failed for {url}: {e} — trying direct.")

    profiles_to_try = [IMPERSONATE_PROFILE] + [
        p for p in IMPERSONATE_FALLBACKS if p != IMPERSONATE_PROFILE
    ]

    last_exc = None
    for profile in profiles_to_try:
        try:
            kwargs = {"impersonate": profile, "timeout": REQUEST_TIMEOUT}
            if SCRAPE_PROXY:
                kwargs["proxies"] = {"http": SCRAPE_PROXY, "https": SCRAPE_PROXY}

            response = curl_requests.get(url, **kwargs)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                raise ValueError(f"Skipping non-HTML content ({content_type}) at {url}")

            if profile != IMPERSONATE_PROFILE:
                print(f"[FETCH] {url} succeeded with fallback profile '{profile}'")
            return response.text

        except Exception as e:
            last_exc = e
            is_403 = "403" in str(e)
            print(f"[FETCH] Profile '{profile}' failed for {url}: {e}")
            if not is_403:
                break

    raise last_exc


def _prioritize_urls(urls):
    """Sorts URLs so 'core' pages come before low-priority ones (blog posts
    etc). Decides both what survives the MAX_PAGES_CEILING cut and what gets
    visited first — which is why the crawl queue below is an ordered deque
    rather than a set."""
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

    # Floor the budget at MAX_PAGES: a small sitemap (say 12 URLs) shouldn't
    # leave less room than a site with no sitemap at all, or internal links
    # discovered while crawling could never be followed.
    effective = max(min(len(ordered), MAX_PAGES_CEILING), MAX_PAGES)

    print(f"[SITEMAP] {len(ordered)} URLs declared for {main_url}"
          + (f", capped to {MAX_PAGES_CEILING} (low-priority pages dropped first)" if capped else "")
          + f". Page budget: {effective}.")

    return ordered[:MAX_PAGES_CEILING], effective


def _extract_internal_links(url, soup):
    """Link extraction off the soup we already have — the page is fetched
    exactly once per URL."""
    links = set()
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        if not is_internal_link(url, href):
            continue
        links.add(urljoin(url, href.split("#")[0]))
    return list(links)


def _extract_tables(soup):
    """Returns a list of row-lists. Includes <th> so Gemini actually sees the
    column headers — without them, rate/fee table summaries lose their labels."""
    tables = []
    for table in soup.find_all("table"):
        table_data = []
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            col_text = [c.get_text(strip=True) for c in cells]
            if any(col_text):
                table_data.append(col_text)
        if table_data:
            tables.append(table_data)
    return tables


def _extract_faqs(soup):
    """Pairs each question with the answer found *inside its own accordion row*,
    instead of zipping two independently-selected lists — one extra toggle or
    missing body in the page would otherwise misalign every pair after it."""
    faqs = []
    faq_container = soup.select_one(".faqs.aem-GridColumn.aem-GridColumn--default--12")
    if not faq_container:
        return faqs

    answer_selector = ".accordion_body, .accordionbody_links, .aem-rte-content"

    for row in faq_container.select(".accordion_row"):
        question_el = row.select_one(".accordion_toggle") or row.select_one("h3, h4, button")
        answer_el = row.select_one(answer_selector)
        if not question_el or not answer_el:
            continue
        question = question_el.get_text(strip=True)
        answer = answer_el.get_text(strip=True)
        if question and answer:
            faqs.append({"question": question, "answer": answer})

    return faqs


async def scrape_page(url):
    """Returns a dict with text, table summaries, FAQs and internal links.
    Links live in the result so a cache hit doesn't trigger a second fetch."""
    global _cache_dirty

    url_hash = hash_url(url)
    if url_hash in scraped_cache:
        print(f"[CACHE] Using cached data for {url}")
        return scraped_cache[url_hash]

    print(f"[SCRAPE] Scraping {url}")
    try:
        html = await asyncio.to_thread(_fetch_html, url)
        soup = BeautifulSoup(html, "lxml")

        # Tables → one batched Gemini summary for the whole page
        tables = _extract_tables(soup)
        table_data_list = []
        if tables:
            try:
                table_data_list.append(await convert_tables_to_sentences_gemini(tables))
            except GeminiQuotaExhausted:
                # Already logged once, loudly. Don't repeat it per page.
                table_data_list.append(_raw_table_fallback(tables))
            except Exception as e:
                print(f"[WARN] Table summaries failed for {url}: {e}")
                table_data_list.append(_raw_table_fallback(tables))

        body = soup.find("body")
        full_text = body.get_text(separator="\n", strip=True) if body else soup.get_text(separator="\n", strip=True)

        try:
            faqs = _extract_faqs(soup)
        except Exception as e:
            print(f"[DEBUG] FAQ extraction failed for {url}: {e}")
            faqs = []

        try:
            links = _extract_internal_links(url, soup)
        except Exception as e:
            print(f"[LINKS] Failed to extract internal links from {url}: {e}")
            links = []

        result = {
            "url": url,
            "full_text": full_text.strip(),
            "table_summaries": table_data_list,
            "faqs": faqs,
            "links": links,
        }

        scraped_cache[url_hash] = result
        _cache_dirty += 1
        _flush_cache()

        print("\n================= [SCRAPED PAGE SUMMARY] =================")
        print(f"URL              : {url}")
        print(f"Text Length      : {len(full_text.split())} words")
        print(f"Tables Found     : {len(table_data_list)}")
        print(f"FAQs Found       : {len(faqs)}")
        print(f"Links Found      : {len(links)}")
        print("Text Preview     :")
        print(full_text.strip()[:300].replace("\n", " ") + "...")
        print("===========================================================\n")

        print(f"[SCRAPE DONE] {url}")
        return result

    except Exception as e:
        print(f"[ERROR] Failed to scrape {url}: {e}")
        return None


async def scrape_site_recursive(main_url):
    main_url = normalize_url(main_url)

    seed_urls, effective_max_pages = get_effective_crawl_plan(main_url)

    # Ordered queue, not a set: _prioritize_urls' ordering has to survive into
    # the crawl, and set.pop() returns an arbitrary element.
    queue = deque()
    queued = set()

    def enqueue(url):
        norm = normalize_url(url)
        if norm not in queued and norm not in scraped:
            queue.append(norm)
            queued.add(norm)

    scraped = set()
    results = []
    failed_urls = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    for u in (seed_urls or [main_url]):
        enqueue(u)

    async def process(url):
        async with semaphore:
            print(f"[VISIT] {url}")
            result = await scrape_page(url)
            links = result.get("links", []) if result else []
            return url, result, links

    while queue and len(scraped) < effective_max_pages:
        remaining_budget = effective_max_pages - len(scraped)
        batch = [queue.popleft() for _ in range(min(len(queue), remaining_budget))]

        print(f"[BATCH] Fetching {len(batch)} pages (scraped so far: {len(scraped)}/{effective_max_pages})")
        batch_results = await asyncio.gather(*(process(url) for url in batch))

        for url, result, links in batch_results:
            scraped.add(url)
            if result:
                results.append(result)
            else:
                failed_urls.append(url)
            for link in links:
                enqueue(link)

    _flush_cache(force=True)

    # Total unique URLs this run ever knew about: the ones it attempted
    # plus whatever was still queued when the page budget ran out.
    total_discovered = len(scraped) + len(queue)

    if len(scraped) >= effective_max_pages and queue:
        print(f"[CRAWL] Hit effective_max_pages={effective_max_pages} cap — {len(queue)} more discovered URLs were left unvisited. "
              f"Raise SCRAPE_MAX_PAGES_CEILING if you need a deeper crawl.")

    _print_crawl_summary(main_url, results, total_discovered, len(scraped), failed_urls)
    return results


# Back-compat alias for existing callers/imports.
scrape_with_playwright_recursive = scrape_site_recursive


def _print_crawl_summary(main_url, results, total_discovered, total_attempted, failed_urls):
    """Prints one consolidated summary of everything scraped in this run,
    so it's easy to see at a glance whether the crawl actually pulled
    real content or just bot-check / empty pages."""
    total_pages = len(results)
    total_words = sum(len(r.get("full_text", "").split()) for r in results)
    total_tables = sum(len(r.get("table_summaries", [])) for r in results)
    total_faqs = sum(len(r.get("faqs", [])) for r in results)
    total_failed = len(failed_urls)

    print("\n" + "=" * 70, flush=True)
    print(f"[CRAWL SUMMARY] Root URL: {main_url}")
    print(f"URLs discovered : {total_discovered}")
    print(f"URLs attempted  : {total_attempted}")
    print(f"Succeeded       : {total_pages}")
    print(f"Failed          : {total_failed}")
    print(f"Total words     : {total_words}")
    print(f"Total tables    : {total_tables}")
    print(f"Total FAQs      : {total_faqs}")
    print("-" * 70, flush=True)

    if not results:
        hint = (
            "no SCRAPER_API_KEY configured — set one to route fetches through a "
            "challenge-solving scraping API" if not SCRAPER_API_KEY else
            "scraping API fetch also failed — check SCRAPER_API_KEY / provider quota"
        )
        print(f"⚠️  No pages were scraped — likely Cloudflare bot-blocking ({hint}).")

    for i, r in enumerate(results, start=1):
        word_count = len(r.get("full_text", "").split())
        table_count = len(r.get("table_summaries", []))
        faq_count = len(r.get("faqs", []))
        flag = " ⚠️ low content" if word_count < 50 else ""
        print(f"{i:>3}. {r.get('url')}")
        print(f"     words={word_count}  tables={table_count}  faqs={faq_count}{flag}")

    if failed_urls:
        print("-" * 70)
        print(f"FAILED URLS ({total_failed}):")
        for i, url in enumerate(failed_urls, start=1):
            print(f"{i:>3}. {url}")

    print("=" * 70 + "\n")


class GeminiQuotaExhausted(RuntimeError):
    """Raised when the per-day free-tier quota is gone — not retryable today."""


def _is_429(err_text):
    return "429" in err_text or "RESOURCE_EXHAUSTED" in err_text


def _is_daily_quota(err_text):
    """Daily project quota vs per-minute rate limit. The quotaId carries it:
    GenerateRequestsPerDayPerProjectPerModel-FreeTier."""
    return "PerDay" in err_text or "FreeTier" in err_text and "per day" in err_text.lower()


def _retry_delay_from_error(err_text, default):
    """The API returns the exact wait in RetryInfo ('retryDelay': '48s') and in
    the message ('Please retry in 48.000971238s'). Use it instead of guessing."""
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_text) or \
        re.search(r"'retryDelay': '(\d+(?:\.\d+)?)s'", err_text)
    if match:
        return min(float(match.group(1)) + 1.0, 120.0)
    return default


async def _gemini_rate_limited_call(prompt):
    """Serializes Gemini calls and enforces GEMINI_RPM spacing between them.
    Per-minute 429s are retried after the server-supplied delay; a daily quota
    429 trips the circuit breaker so the rest of the run skips Gemini entirely.
    Backoff sleeps happen *outside* the lock — holding it would stall every
    other pending call for the full backoff."""
    global _gemini_last_call, _gemini_disabled, _gemini_disabled_reason

    if _gemini_disabled:
        raise GeminiQuotaExhausted(_gemini_disabled_reason)

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            async with _gemini_lock:
                if _gemini_disabled:  # another coroutine tripped it while we waited
                    raise GeminiQuotaExhausted(_gemini_disabled_reason)
                wait = _gemini_min_interval - (time.monotonic() - _gemini_last_call)
                if wait > 0:
                    await asyncio.sleep(wait)
                try:
                    response = await asyncio.to_thread(
                        genai_client.models.generate_content,
                        model=GEMINI_MODEL,
                        contents=prompt,
                    )
                    return response.text
                finally:
                    _gemini_last_call = time.monotonic()
        except GeminiQuotaExhausted:
            raise
        except Exception as e:
            err_text = str(e)
            if not _is_429(err_text):
                raise

            if _is_daily_quota(err_text):
                _gemini_disabled = True
                _gemini_disabled_reason = (
                    f"daily free-tier quota for {GEMINI_MODEL} exhausted — resets at "
                    f"midnight US/Pacific"
                )
                print(f"\n[GEMINI] ⛔ {_gemini_disabled_reason}. Skipping all further "
                      f"table summaries this run; raw table text will be used instead. "
                      f"Enable billing or set GEMINI_MODEL to a model with headroom.\n")
                raise GeminiQuotaExhausted(_gemini_disabled_reason) from e

            if attempt < max_retries:
                backoff = _retry_delay_from_error(err_text, default=20 * (attempt + 1))
                print(f"[GEMINI] Per-minute rate limit, retrying in {backoff:.0f}s "
                      f"(attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(backoff)
                continue
            raise


def _render_tables(tables):
    """Flattens a page's tables into one prompt-ready block."""
    blocks = []
    for i, table_data in enumerate(tables, start=1):
        rows = "\n".join(", ".join(row) for row in table_data)
        blocks.append(f"Table {i}:\n{rows}")
    return "\n\n".join(blocks)


def _raw_table_fallback(tables):
    return "\n\n".join(
        f"[Table {i} - raw, summary unavailable] "
        + "; ".join(", ".join(row) for row in table_data)
        for i, table_data in enumerate(tables, start=1)
    )


async def convert_tables_to_sentences_gemini(tables):
    """One call per PAGE, not per table — the service-fee page alone has 4+
    tables, and at 20 requests/day that difference is the whole budget.
    Cached on the tables' content hash so repeated tables cost nothing."""
    global _cache_dirty

    if not tables:
        return ""

    rendered = _render_tables(tables)
    key = hashlib.sha256(rendered.encode("utf-8")).hexdigest()

    if key in table_summary_cache:
        print(f"[CACHE] Using cached summary for {len(tables)} table(s)")
        return table_summary_cache[key]

    prompt = (
        "Convert each of the following tables into descriptive sentences. "
        "Keep the 'Table N:' heading above each one's sentences.\n\n"
        f"{rendered}"
    )
    summary = await _gemini_rate_limited_call(prompt)

    table_summary_cache[key] = summary
    with open(TABLE_CACHE_PATH, "wb") as f:
        pickle.dump(table_summary_cache, f)

    return summary


async def convert_table_to_sentences_gemini(table_data, index):
    """Kept for back-compat with any caller that still summarises one table."""
    return await convert_tables_to_sentences_gemini([table_data])


async def scrape_web_data(url):
    results = await scrape_site_recursive(url)
    if not results:
        hint = (
            "Set SCRAPER_API_KEY (see webscrape.py) to route fetches through a "
            "challenge-solving scraping API." if not SCRAPER_API_KEY else
            "Scraping API fetch failed too — check SCRAPER_API_KEY / provider quota/credits."
        )
        raise RuntimeError(
            f"Scraping {url} returned 0 pages — the site is very likely blocking direct "
            f"requests with a Cloudflare interactive challenge. {hint}"
        )
    return results