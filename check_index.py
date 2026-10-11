import os
import pickle
import re

PROJECT = "local-test"
INDEX_DIR = os.path.join("projects", PROJECT, "faiss_index")

with open(os.path.join(INDEX_DIR, "index.pkl"), "rb") as f:
    data = pickle.load(f)  # Only load your own trusted pickle files.

docstore = data["docstore"]
documents = list(docstore._dict.values())

print("Total indexed chunks:", len(documents))

patterns = [
    r"7\.90\s*%",
    r"7\.9\s*%",
    r"30\s*months",
    r"36\s*months",
]

for pattern in patterns:
    matches = [
        doc for doc in documents
        if re.search(pattern, doc.page_content, re.IGNORECASE)
    ]

    print(f"\nPATTERN: {pattern}")
    print(f"MATCHES: {len(matches)}")

    for doc in matches[:5]:
        print("SOURCE:", doc.metadata)
        print("TEXT:", doc.page_content[:1200])
        print("-" * 60)