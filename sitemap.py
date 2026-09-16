import asyncio
import os
from curl_cffi import requests as curl_requests
from lxml import etree
from urllib.parse import urljoin

# Same env var webscrape.py reads, so both files stay in sync on which
# browser fingerprint curl_cffi impersonates.
IMPERSONATE_PROFILE = os.getenv("SCRAPE_IMPERSONATE", "chrome119")
REQUEST_TIMEOUT = 20

# NEW: outbound proxy for scraping. Set this in your deployment env
# (docker-compose / systemd / ECS task def), NOT locally — this is the
# actual fix for "works on my Mac, 403s on the server". AWS/GCP/Azure
# datacenter IPs are blocked or hard-challenged by Cloudflare/Akamai-class
# WAFs on banking sites independent of TLS fingerprint. Use a residential
# or ISP-tier proxy provider (e.g. Bright Data, Oxylabs, Smartproxy).
# Format: "http://user:pass@proxy-host:port"
SCRAPE_PROXY = os.getenv("SCRAPE_PROXY")  # leave unset to disable

# Fallback chain of impersonation profiles to retry on 403, in case the
# pinned profile itself has also gone stale (Cloudflare periodically
# invalidates specific JA3 fingerprints).
IMPERSONATE_FALLBACKS = [
    p.strip() for p in os.getenv(
        "SCRAPE_IMPERSONATE_FALLBACKS",
        "chrome119,chrome124,chrome120,chrome131,safari17_0"
    ).split(",") if p.strip()
]

# Common sitemap locations to try, in order, before giving up
SITEMAP_CANDIDATES = ["/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml"]


def _fetch(url):
    """Fetches a URL, trying the pinned impersonate profile first, then
    falling back through IMPERSONATE_FALLBACKS on 403 (Forbidden) —
    covers the case where Cloudflare has invalidated one fingerprint but
    not others. Routes through SCRAPE_PROXY if set."""
    profiles_to_try = [IMPERSONATE_PROFILE] + [
        p for p in IMPERSONATE_FALLBACKS if p != IMPERSONATE_PROFILE
    ]

    last_exc = None
    for profile in profiles_to_try:
        try:
            kwargs = {
                "impersonate": profile,
                "timeout": REQUEST_TIMEOUT,
            }
            if SCRAPE_PROXY:
                kwargs["proxies"] = {"http": SCRAPE_PROXY, "https": SCRAPE_PROXY}

            response = curl_requests.get(url, **kwargs)
            response.raise_for_status()

            if profile != IMPERSONATE_PROFILE:
                print(f"[FETCH] {url} succeeded with fallback profile '{profile}' "
                      f"(pinned '{IMPERSONATE_PROFILE}' was blocked)")
            return response.content

        except Exception as e:
            last_exc = e
            is_403 = "403" in str(e)
            print(f"[FETCH] Profile '{profile}' failed for {url}: {e}")
            if not is_403:
                # Non-403 errors (timeout, DNS, etc.) won't be fixed by
                # switching TLS fingerprint — stop trying other profiles.
                break

    raise last_exc


def _get_sitemap_urls_from_robots(base_url):
    """robots.txt often lists the real sitemap path via 'Sitemap: <url>' lines —
    more reliable than guessing /sitemap.xml, since some sites use a different path."""
    try:
        robots = _fetch(urljoin(base_url, "/robots.txt")).decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"[ROBOTS] Couldn't fetch robots.txt: {e}")
        return []

    sitemap_urls = []
    for line in robots.splitlines():
        line = line.strip()
        if line.lower().startswith("sitemap:"):
            sitemap_urls.append(line.split(":", 1)[1].strip())
    return sitemap_urls


def _parse_sitemap_xml(xml_bytes):
    """Returns (kind, entries) where kind is 'index' (list of sub-sitemap URLs)
    or 'urlset' (list of page URLs)."""
    root = etree.fromstring(xml_bytes)
    tag = etree.QName(root).localname

    ns = {"sm": root.nsmap.get(None, "")} if root.nsmap.get(None) else {}
    loc_path = ".//sm:loc" if ns else ".//loc"

    locs = [el.text.strip() for el in root.findall(loc_path, ns) if el.text]
    if tag == "sitemapindex":
        return "index", locs
    return "urlset", locs


def discover_all_urls(base_url, max_sub_sitemaps=200, verbose=True):
    """Walks sitemap index -> sub-sitemaps -> page URLs, returns the full set
    of unique page URLs the site's sitemap declares."""
    base_url = base_url.rstrip("/")
    all_page_urls = set()

    candidates = _get_sitemap_urls_from_robots(base_url)
    if not candidates:
        candidates = [urljoin(base_url + "/", path.lstrip("/")) for path in SITEMAP_CANDIDATES]

    to_visit = list(dict.fromkeys(candidates))
    visited_sitemaps = set()

    while to_visit and len(visited_sitemaps) < max_sub_sitemaps:
        sitemap_url = to_visit.pop(0)
        if sitemap_url in visited_sitemaps:
            continue
        visited_sitemaps.add(sitemap_url)

        try:
            xml_bytes = _fetch(sitemap_url)
            kind, locs = _parse_sitemap_xml(xml_bytes)
        except Exception as e:
            print(f"[SITEMAP] Failed to fetch/parse {sitemap_url}: {e}")
            continue

        if kind == "index":
            if verbose:
                print(f"[SITEMAP INDEX] {sitemap_url} -> {len(locs)} sub-sitemaps")
            to_visit.extend(loc for loc in locs if loc not in visited_sitemaps)
        else:
            if verbose:
                print(f"[SITEMAP] {sitemap_url} -> {len(locs)} URLs")
            all_page_urls.update(locs)

    if not all_page_urls:
        print("⚠️  No sitemap found or it returned zero URLs — site may not expose one, "
              "or it's behind the same bot protection. Falls back to link-crawling instead.")

    return all_page_urls


if __name__ == "__main__":
    urls = discover_all_urls("https://www.au.bank.in")
    print(f"\nTotal unique URLs found: {len(urls)}")