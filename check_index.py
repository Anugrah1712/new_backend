
import os
import pickle

PROJECT = "local-test"
INDEX_DIR = os.path.join("projects", PROJECT, "faiss_index")

TARGET_URL = (
    "https://www.au.bank.in/interest-rates/"
    "savings-account-interest-rates"
)

with open(os.path.join(INDEX_DIR, "index.pkl"), "rb") as f:
    data = pickle.load(f)  # Only load trusted pickle files.

docstore = data["docstore"]
documents = list(docstore._dict.values())

print(f"\nTotal indexed chunks: {len(documents)}")
print(f"Searching for URL: {TARGET_URL}\n")


def normalize_url(url):
    return str(url).strip().rstrip("/").lower()


target = normalize_url(TARGET_URL)

matches = []

for index, doc in enumerate(documents):
    metadata = doc.metadata or {}

    # URLs can be stored in different metadata fields
    urls = []

    for key in ("source_urls", "source_url", "url"):
        value = metadata.get(key, [])

        if isinstance(value, str):
            urls.append(value)
        elif isinstance(value, (list, tuple, set)):
            urls.extend(value)

    # Also check URL embedded in chunk content
    first_lines = doc.page_content[:500]

    metadata_match = any(
        normalize_url(url) == target
        for url in urls
    )

    content_match = any(
        normalize_url(line.removeprefix("URL:").strip()) == target
        for line in first_lines.splitlines()
        if line.strip().lower().startswith("url:")
    )

    if metadata_match or content_match:
        matches.append((index, doc))


print("=" * 90)
print(f"TOTAL CHUNKS FOUND FOR TARGET URL: {len(matches)}")
print("=" * 90)

for chunk_no, (index, doc) in enumerate(matches, start=1):
    print(f"\n{'=' * 90}")
    print(f"CHUNK NUMBER: {chunk_no}")
    print(f"DOCSTORE POSITION: {index}")
    print(f"CHARACTER LENGTH: {len(doc.page_content)}")
    print(f"METADATA: {doc.metadata}")
    print("-" * 90)
    print(doc.page_content)
    print("=" * 90)

if not matches:
    print("\nNo chunks found for the specified URL.")
    print("The page may be absent from the index,")
    print("or its chunks may not retain the URL in metadata.")
