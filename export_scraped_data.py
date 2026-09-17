# export_scraped_data.py
"""
Exports a scrape cache (scraped_cache.pkl) into human-readable files so you
can actually look at what got scraped. Run standalone:

    python export_scraped_data.py                          # uses ./scraped_cache.pkl
    python export_scraped_data.py path/to/scraped_cache.pkl # explicit path
    python export_scraped_data.py --project au-bank-demo    # uses projects/<name>/scraped_cache.pkl

Writes two files next to the cache:
    <name>_export.md    — human-readable, one section per page
    <name>_export.json  — same data, structured, for scripting/diffing
"""
import json
import os
import pickle
import sys

BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", os.path.join(os.path.dirname(__file__), "projects"))


def resolve_cache_path():
    args = sys.argv[1:]

    if "--project" in args:
        idx = args.index("--project")
        project_name = args[idx + 1]
        return os.path.join(BASE_OUTPUT_DIR, project_name, "scraped_cache.pkl")

    if args:
        return args[0]

    return "scraped_cache.pkl"


def load_cache(cache_path):
    if not os.path.exists(cache_path):
        print(f"❌ No cache found at: {cache_path}")
        sys.exit(1)

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    # scraped_cache.pkl is {url_hash: result_dict}; the domain-folder
    # scraped_cache.pkl (written by main.py's /preprocess) is a plain list
    # of result_dicts instead — normalize both to a list here.
    if isinstance(data, dict):
        return list(data.values())
    return data


def export(cache_path):
    results = load_cache(cache_path)
    base = os.path.splitext(cache_path)[0]
    md_path = f"{base}_export.md"
    json_path = f"{base}_export.json"

    # --- JSON (structured, exact) ---
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # --- Markdown (readable) ---
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Scraped Data Export\n\n")
        f.write(f"Source: `{cache_path}`  \n")
        f.write(f"Total pages: {len(results)}\n\n")
        f.write("---\n\n")

        for i, r in enumerate(results, start=1):
            if not r:
                continue
            url = r.get("url", "(no url)")
            full_text = r.get("full_text", "")
            tables = r.get("table_summaries", [])
            faqs = r.get("faqs", [])

            f.write(f"## {i}. {url}\n\n")
            f.write(f"**Words:** {len(full_text.split())}  |  "
                    f"**Tables:** {len(tables)}  |  **FAQs:** {len(faqs)}\n\n")

            if faqs:
                f.write("### FAQs\n\n")
                for faq in faqs:
                    f.write(f"- **Q:** {faq.get('question', '')}\n")
                    f.write(f"  **A:** {faq.get('answer', '')}\n")
                f.write("\n")

            if tables:
                f.write("### Table Summaries\n\n")
                for t in tables:
                    f.write(f"{t}\n\n")

            f.write("### Full Text\n\n")
            f.write("```\n")
            f.write(full_text)
            f.write("\n```\n\n")
            f.write("---\n\n")

    print(f"✅ Exported {len(results)} pages")
    print(f"   Readable: {md_path}")
    print(f"   Structured: {json_path}")


if __name__ == "__main__":
    export(resolve_cache_path())