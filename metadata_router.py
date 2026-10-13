
# metadata_router.py

import json
import re
from functools import lru_cache
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_url(url):
    p = urlsplit(str(url).strip())
    return urlunsplit((
        p.scheme.lower(),
        p.netloc.lower(),
        p.path.rstrip("/") or "/",
        "",
        ""
    ))


def normalize_product(text):
    return re.sub(
        r"[^a-z0-9]+", " ",
        str(text).lower().replace("_", " ").replace("-", " ")
    ).strip()


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list) or not records:
        raise ValueError("Metadata must be a non-empty JSON array.")

    result = []
    seen = set()

    for i, item in enumerate(records, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Invalid record {i}")

        if not all(k in item for k in ("url", "category", "product", "summary")):
            raise ValueError(f"Missing metadata fields in record {i}")

        url = normalize_url(item["url"])
        products = item["product"]

        if (
            not url.startswith(("https://", "http://"))
            or not isinstance(products, list)
            or not all(isinstance(p, str) and p.strip() for p in products)
            or not isinstance(item["summary"], str)
        ):
            raise ValueError(f"Invalid metadata fields in record {i}")

        if url in seen:
            raise ValueError(f"Duplicate metadata URL: {url}")

        seen.add(url)
        result.append({
            "url": url,
            "category": str(item["category"]),
            "product": list(dict.fromkeys(products)),
            "summary": item["summary"].strip()
        })

    return result


def distinct_products(records):
    return sorted({
        product
        for item in records
        for product in item["product"]
    })


def _aliases(product):
    name = normalize_product(product)
    aliases = {name}

    # Generic abbreviations, never additional product identifiers.
    if name == "fixed deposit":
        aliases.update(("fd", "fds", "fixed deposits", "term deposit"))
    if name == "savings account":
        aliases.update(("savings", "saving account", "savings accounts"))
    if name == "current account":
        aliases.add("current accounts")

    # Allow shortened multiword product names only when the abbreviation
    # identifies a single metadata product (checked by resolve_product).
    tokens = name.split()
    if len(tokens) > 1 and tokens[-1] in {"loan", "loans", "account", "accounts", "card", "cards", "deposit", "deposits"}:
        short = " ".join(tokens[:-1])
        if short and short not in {"personal", "business", "home", "car", "credit", "savings", "current"}:
            aliases.add(short)
    return aliases


def _mentioned(text, phrase):
    return bool(re.search(
        r"(?<!\w)" + re.escape(phrase) + r"(?!\w)",
        text
    ))


def resolve_product(question, products):
    """Prefer exact canonical product names over shortened aliases."""
    query = normalize_product(question)
    matches = []
    for product in products:
        canonical = normalize_product(product)
        for alias in _aliases(product):
            if alias and _mentioned(query, alias):
                # Canonical names outrank aliases at equal phrase length.
                # This avoids `fixed_deposit_account`'s shortened alias
                # `fixed deposit` competing with actual `fixed_deposit`.
                matches.append((len(alias.split()), len(alias),
                                int(alias == canonical), product))
    if not matches:
        return None, "missing"
    best = max((m[0], m[1], m[2]) for m in matches)
    winners = {m[3] for m in matches if m[:3] == best}
    if len(winners) != 1:
        return None, "ambiguous"
    return next(iter(winners)), "matched"


def _product_family(product):
    """Classify supported metadata products without requiring a fixed product list."""
    name = normalize_product(product)
    if re.search(r"\b(loans?|overdraft|credit line)\b", name):
        return "loan"
    if re.search(r"\b(accounts?|savings|current)\b", name):
        return "account"
    if re.search(r"\b(deposits?|fd|rd)\b", name):
        return "deposit"
    if re.search(r"\b(cards?)\b", name):
        return "card"
    return None


def _requested_family(question):
    q = normalize_product(question)
    if re.search(r"\b(loans?|borrowing)\b", q):
        return "loan"
    if re.search(r"\b(accounts?)\b", q):
        return "account"
    if re.search(r"\b(deposits?|fds?|rds?)\b", q):
        return "deposit"
    if re.search(r"\b(cards?)\b", q):
        return "card"
    return None


def _display_product(product):
    return normalize_product(product).title().replace("Msme", "MSME").replace("Sme", "SME").replace("Nri", "NRI")


def _family_examples(products):
    # Prioritize commonly requested names, but only when actually present.
    preferred = ("personal loan", "home loan", "business loan", "msme loan", "car loan", "two wheeler loan",
                 "savings account", "current account", "fixed deposit")
    return ", ".join(_display_product(p) for p in sorted(products, key=lambda p: (
        next((i for i, phrase in enumerate(preferred) if normalize_product(p) == phrase), len(preferred)),
        normalize_product(p)))[:6])


def _clarify(question, state, message):
    state["pending_question"] = question
    return {"status": "clarify", "response": message, "state": state}


def _looks_like_followup(question):
    q = normalize_product(question)
    # Explicit references and short questions about the current product.
    if re.search(r"\b(it|its|they|their|that|this|same|above)\b", q):
        return True
    if re.match(r"^(and|also|what about|how about|tell me more)\b", q):
        return True
    if len(q.split()) <= 8 and re.search(
        r"\b(rate|rates|interest|documents?|eligibility|eligible|"
        r"charges?|fees?|tenure|benefits?|features?|limit|limits|"
        r"apply|application|requirements?|minimum|maximum)\b", q
    ):
        return True
    return False


def is_product_clarification(question):
    q = normalize_product(question)
    return bool(re.search(
        r"\b(interest|rates?|charges?|fees?|eligib(?:le|ility)|"
        r"documents?|tenure|benefits?|features?|apply|application|"
        r"limit|limits|open|opening|close|closing|requirements?|"
        r"minimum|maximum|how|what|can|could|want|need)\b", q
    ))


def route_question(question, records, conversation_state=None):
    """Resolve product and follow-ups before retrieval, asking on ambiguity.

    Explicit product > current-family continuation > category clarification >
    genuine contextual follow-up > general product clarification.
    """
    state = dict(conversation_state or {})
    products = distinct_products(records)
    product, status = resolve_product(question, products)
    previous = state.get("product")
    pending = state.get("pending_question")
    family = _requested_family(question)

    # Generic family labels (e.g. "loan" or "account") are not specific
    # products, even if the metadata includes a product with that exact name.
    # Handle them BEFORE accepting resolve_product()'s matched result.
    generic_labels = {
        "loan": {"loan", "loans", "borrowing"},
        "account": {"account", "accounts"},
        "deposit": {"deposit", "deposits"},
        "card": {"card", "cards"},
    }
    q = normalize_product(question)
    if family and any(_mentioned(q, label) for label in generic_labels[family]):
        explicit_specific = (
            status == "matched" and product is not None
            and normalize_product(product) not in generic_labels[family]
            and any(
                _mentioned(q, alias)
                for alias in _aliases(product)
                if alias not in generic_labels[family]
            )
        )
        if not explicit_specific:
            family_products = [
                p for p in products
                if _product_family(p) == family
                and normalize_product(p) not in generic_labels[family]
            ]
            # Only inherit context if it is a specific product in this family.
            if previous in family_products and not pending:
                return {"status": "ready", "product": previous,
                        "question": question, "state": state}
            if len(family_products) == 1:
                selected = family_products[0]
                state["product"] = selected
                state["pending_question"] = None
                return {"status": "ready", "product": selected,
                        "question": question, "state": state}
            state["product"] = None
            options = _family_examples(family_products)
            message = f"Which type of {family} are you interested in?"
            if options:
                message += f" For example: {options}."
            return _clarify(question, state, message)

    if status == "matched":
        # Only combine a pending question if the reply is actually resolving it.
        if (pending and len(normalize_product(question).split()) <= 5
                and (not _requested_family(pending)
                     or _requested_family(pending) == _product_family(product))):
            full_question = pending + "\nProduct: " + product
        else:
            full_question = question
        state["pending_question"] = None
        state["product"] = product
        return {"status": "ready", "product": product,
                "question": full_question, "state": state}

    if status == "ambiguous":
        return _clarify(question, state, "Which specific product do you mean?")

    if family:
        family_products = [p for p in products if _product_family(p) == family]
        # Follow-up within the same family can retain the previous selection.
        if previous in family_products and not pending:
            return {"status": "ready", "product": previous,
                    "question": question, "state": state}
        if len(family_products) == 1:
            selected = family_products[0]
            state["product"] = selected
            state["pending_question"] = None
            return {"status": "ready", "product": selected,
                    "question": question, "state": state}
        if len(family_products) > 1:
            # Show actual products from the uploaded metadata, not invented names.
            options = _family_examples(family_products)
            return _clarify(question, state,
                f"Which {family} product are you referring to? For example: {options}.")
        return _clarify(question, state,
            f"Which specific {family} product are you referring to?")

    # If the bot asked for a product and the reply did not match metadata,
    # keep asking rather than silently falling back to an old product.
    if pending:
        return _clarify(pending, state,
            "Please specify the product name so I can answer accurately.")

    if previous and _looks_like_followup(question):
        return {"status": "ready", "product": previous,
                "question": question, "state": state}

    if is_product_clarification(question):
        return _clarify(question, state, "Which product are you referring to?")

    return {"status": "unsupported", "response": (
        "Sorry, I don't have information about that product "
        "in the available knowledge base."), "state": state}


def rank_metadata_urls(question, product, records, embedding_model,
                       limit=12, indexed_urls=None):
    available = (
        {normalize_url(url) for url in indexed_urls}
        if indexed_urls is not None else None
    )
    candidates = [
        item for item in records
        if product in item["product"]
        and (available is None or normalize_url(item["url"]) in available)
    ]

    if not candidates:
        return []

    texts = [
        " ".join((
            item["url"].replace("-", " ").replace("/", " "),
            item["category"],
            " ".join(item["product"]).replace("_", " "),
            item["summary"]
        ))
        for item in candidates
    ]

    query = question + " " + normalize_product(product)

    corpus = [query] + texts
    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    lexical_matrix = tfidf.fit_transform(corpus)
    lexical = cosine_similarity(
        lexical_matrix[0:1], lexical_matrix[1:]
    ).ravel()

    # Use the same embedding model as the content FAISS index.
    q_vec = np.asarray(
        embedding_model.embed_query(query), dtype=np.float32
    ).reshape(1, -1)
    doc_vecs = np.asarray(
        embedding_model.embed_documents(texts), dtype=np.float32
    )

    semantic = cosine_similarity(q_vec, doc_vecs).ravel()
    scores = 0.45 * lexical + 0.55 * semantic

    # Prefer core product/rate pages over incidental mentions in blogs.
    for i, item in enumerate(candidates):
        path = urlsplit(item["url"]).path.lower()
        if item["category"] == "main":
            scores[i] += 0.04
        if "interest" in query.lower() and "interest-rate" in path:
            scores[i] += 0.10

    positions = np.argsort(-scores)[:limit]

    return [
        {**candidates[int(i)], "score": float(scores[int(i)])}
        for i in positions
    ]


def select_urls_with_llm(question, product, candidates, chat_model,
                         max_urls=3):
    if not candidates:
        return []

    # Reuse the existing model handler; validate its output strictly.
    from inference import run_chat_model

    shortlist = [
        {
            "id": i,
            "url": c["url"],
            "category": c["category"],
            "summary": c["summary"][:1400]
        }
        for i, c in enumerate(candidates)
    ]

    instruction = """
Select the best source URLs for answering the user question.

Return ONLY JSON:
{"ids": [0, 1]}

Rules:
- Select up to 3 candidate IDs.
- Use only IDs in the supplied list.
- Prefer the authoritative page directly answering the question.
- Prefer regular/main rate pages for generic rate questions.
- Do not select green/bulk/special variants unless asked.
- Do not invent URLs, facts, products, or IDs.
- If no candidate is relevant, return {"ids": []}.
"""

    context = json.dumps(shortlist, ensure_ascii=False)
    response = run_chat_model(
        chat_model=chat_model,
        context=context,
        question=f"Product: {product}\nQuestion: {question}",
        chat_history=[],
        custom_instructions=instruction,
        max_output_tokens=400,
        temperature=0
    )

    allowed = {i: c["url"] for i, c in enumerate(candidates)}

    try:
        match = re.search(r"\{[\s\S]*\}", response)
        payload = json.loads(match.group(0)) if match else {}
        ids = payload.get("ids", [])

        if not isinstance(ids, list):
            ids = []

        selected = []
        for i in ids:
            if type(i) is int and i in allowed:
                url = allowed[i]
                if url not in selected:
                    selected.append(url)
            if len(selected) >= max_urls:
                break

        return selected
    except (ValueError, TypeError, AttributeError):
        return []
