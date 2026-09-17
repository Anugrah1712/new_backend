import copy
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
from sitemap import discover_all_urls

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
#
# Default is safari17_0, not a Chrome profile: on au.bank.in, every single
# fetch failed on chrome119 (and usually chrome124 too) before succeeding on
# safari17_0 or chrome120 — this specific WAF has that fingerprint flagged.
# Starting with the profile that actually wins cuts 2-4 wasted requests per
# page down to ~0-1.
IMPERSONATE_PROFILE = os.getenv("SCRAPE_IMPERSONATE", "safari17_0")

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
MAX_PAGES_CEILING = int(os.getenv("SCRAPE_MAX_PAGES_CEILING", "3350"))

# URL substrings that mark "lower priority" content — pages containing any
# of these get pushed to the end of the queue and are the first to be cut
# when the sitemap total exceeds MAX_PAGES_CEILING. Extend as needed
# (e.g. "/press-release/", "/careers/").
LOW_PRIORITY_PATTERNS = ["/blogs/", "/campaign/"]

MAX_CONCURRENT_REQUESTS = int(os.getenv("SCRAPE_CONCURRENCY", "5"))

# Class-name substrings worth trying opportunistically for the per-page
# strip below, even though we don't have this site's confirmed markup (same
# blind spot as the FAQ selector saga — we only ever see rendered/markdown
# output, never raw HTML with real class names). Harmless if nothing
# matches; the safety net in _extract_clean_text backs off if a hint
# accidentally nukes most of the page.
NAV_FOOTER_CLASS_HINTS = [
    "header", "navbar", "main-nav", "mainnav", "globalnav", "site-header",
    "footer", "site-footer", "mega-menu", "megamenu", "language-selector",
]

# How much of a full crawl's pages a given text LINE has to appear on,
# verbatim, before we treat it as boilerplate (nav/footer/language-picker)
# rather than real content, and strip it from every page. This is the
# primary defense against nav/footer duplication — unlike the class-name
# guesses above, it doesn't need to know anything about this site's actual
# markup: it's driven purely by what's observed to repeat across pages.
BOILERPLATE_FREQUENCY_THRESHOLD = float(os.getenv("SCRAPE_BOILERPLATE_THRESHOLD", "0.5"))
BOILERPLATE_MIN_PAGES = int(os.getenv("SCRAPE_BOILERPLATE_MIN_PAGES", "5"))


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
    """Locates the FAQ section by heading TEXT rather than a hardcoded CSS
    class. The class this used to look for (.faqs.aem-GridColumn.aem-Grid
    Column--default--12) was copied from a different AEM site's build and
    matches nothing on au.bank.in — every page silently reported 0 FAQs. AEM
    sites vary their component class names release to release, but the
    visible heading text ("Frequently Asked Questions") is far more stable,
    so we anchor there and walk the DOM structurally instead of guessing
    another exact class."""
    faqs = []

    heading = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        text = tag.get_text(strip=True).lower()
        if "frequently asked question" in text or text == "faqs" or text == "faq":
            heading = tag
            break

    if heading is None:
        return faqs

    # The FAQ block is typically the heading's parent container (or a level
    # or two up) holding a repeated list of question/answer units. Walk up
    # to find something wide enough to hold the whole list, capped so we
    # don't accidentally grab the <body>.
    container = heading.parent
    for _ in range(3):
        if container is None or container.name == "body":
            break
        if len(container.find_all(["h3", "h4", "button", "summary"])) >= 2:
            break
        container = container.parent

    if container is None:
        return faqs

    # Candidate question elements: heading-like or accordion toggle/button/
    # summary tags within the container. We don't require matching `heading`
    # itself here — it's often an h1/h2 while questions are h3/h4, so it
    # would never appear in this list and a "have we passed it yet" flag
    # would never flip. Since `container` was already anchored on `heading`,
    # everything found here already sits at-or-after it structurally.
    candidates = [c for c in container.find_all(["h3", "h4", "button", "summary"])
                  if c is not heading]

    for i, q_el in enumerate(candidates):
        question = q_el.get_text(strip=True)
        if not question or len(question) > 300:
            continue  # skip empty toggles or accidental non-question matches

        # The answer is whatever text sits between this question element and
        # the next one in document order — covers <details>, accordion divs,
        # or a plain following sibling, without needing a specific class name.
        answer_parts = []
        for sib in q_el.find_all_next():
            if sib in candidates[i + 1:]:
                break
            if sib.name in ("h3", "h4", "button", "summary"):
                break
            # Only take leaf text nodes (no nested tags). A wrapping <div> or
            # <li> around the *next* question/answer block also matches
            # name+has-text, but its .get_text() pulls in everything nested
            # inside it — including the next Q&A — well before we reach that
            # block's own heading tag in this flattened traversal. Skipping
            # non-leaf containers avoids that bleed.
            if sib.name in ("p", "li", "div", "span") and not sib.find(True) \
                    and sib.get_text(strip=True):
                answer_parts.append(sib.get_text(strip=True))

        answer = " ".join(dict.fromkeys(answer_parts))  # de-dupe, keep order
        # Strip the "See more" expander boilerplate that AEM injects into every card.
        answer = re.sub(r"\bSee more\b\.?$", "", answer, flags=re.IGNORECASE).strip()

        if question and answer:
            faqs.append({"question": question, "answer": answer})

    return faqs


def _extract_clean_text(soup):
    """Extracts body text with nav/header/footer removed first. Cheap, per-
    page pass — a first cut, not the primary defense (see
    _strip_repeated_boilerplate_lines below for that). Operates on a
    deep-copied body so table/FAQ extraction upstream (which runs on the
    original `soup`) is never affected by this.

    Every page of this site's full_text was coming back with the entire nav
    menu and footer duplicated in — the same "Personal Business NRI
    Premium... Language English..." block and the same footer link list on
    every single page, which is what flooded the downstream chunker with
    ~27,000 near-identical chunks. We don't have this site's real HTML
    (only markdown-rendered output — same blind spot as the FAQ selector),
    so this pass tries removing <nav>/<header>/<footer> tags plus a list of
    common nav/footer class-name substrings, with a safety net: if that
    guess nukes more than 80% of the page's text, we assume a hint matched
    something it shouldn't have and use the untouched body instead.
    """
    body = soup.find("body")
    if body is None:
        return soup.get_text(separator="\n", strip=True)

    working = copy.deepcopy(body)
    original_len = len(working.get_text(strip=True))

    for tag_name in ("nav", "header", "footer"):
        for el in working.find_all(tag_name):
            el.decompose()
    for hint in NAV_FOOTER_CLASS_HINTS:
        for el in working.select(f'[class*="{hint}"]'):
            el.decompose()

    stripped_len = len(working.get_text(strip=True))
    if original_len > 0 and stripped_len < original_len * 0.2:
        # Guessed class hints were too aggressive on this page — back off
        # and let the cross-page frequency stripper handle it instead.
        return body.get_text(separator="\n", strip=True)

    return working.get_text(separator="\n", strip=True)


def _strip_repeated_boilerplate_lines(results, min_pages=BOILERPLATE_MIN_PAGES,
                                       frequency_threshold=BOILERPLATE_FREQUENCY_THRESHOLD):
    """Cross-page dedup, run once after the whole crawl. This is the primary
    defense against nav/footer duplication, and unlike _extract_clean_text
    above, it needs no knowledge of this site's actual markup: a text LINE
    that shows up verbatim on a large fraction of pages is, by construction,
    site-wide boilerplate (nav, footer, language picker) rather than page
    content, regardless of what HTML/CSS produced it. Mutates each result's
    full_text in place — since these are the same dict objects stored in
    scraped_cache (scrape_page returns and caches the same object), this
    also updates the cache automatically; the caller flushes it to disk
    afterward.
    """
    if len(results) < min_pages:
        return results  # not enough pages to distinguish content from boilerplate

    line_page_counts = {}
    for r in results:
        # Count each distinct line once per page (not once per occurrence),
        # so a line repeated within a single page's body doesn't inflate
        # its cross-page frequency.
        for line in set(r.get("full_text", "").split("\n")):
            line = line.strip()
            if not line:
                continue
            line_page_counts[line] = line_page_counts.get(line, 0) + 1

    threshold_count = max(3, int(len(results) * frequency_threshold))
    boilerplate_lines = {
        line for line, count in line_page_counts.items()
        if count >= threshold_count
    }

    if not boilerplate_lines:
        return results

    removed_total = 0
    for r in results:
        original_lines = r.get("full_text", "").split("\n")
        kept_lines = [ln for ln in original_lines if ln.strip() not in boilerplate_lines]
        removed_total += len(original_lines) - len(kept_lines)
        r["full_text"] = "\n".join(kept_lines).strip()

    print(f"[BOILERPLATE] Stripped {len(boilerplate_lines)} repeated lines "
          f"(appearing on ≥{int(frequency_threshold * 100)}% of {len(results)} pages) "
          f"— {removed_total} total line removals across the crawl.")

    return results


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

        # Text extraction runs last and on a deep-copied body, so it can't
        # affect the table/FAQ/link extraction above even though it mutates
        # (a copy of) the tree.
        full_text = _extract_clean_text(soup)

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

    # Cross-page boilerplate strip BEFORE flushing, so the on-disk cache
    # reflects the cleaned text too (this mutates the same dict objects that
    # are already stored in scraped_cache — see the function's docstring).
    results = _strip_repeated_boilerplate_lines(results)

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