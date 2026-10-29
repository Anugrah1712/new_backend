
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


# One canonical identifier for each product across routing, state and metadata.
PRODUCT_CANONICAL_MAP = {
    "fd": "fixed_deposit", "fds": "fixed_deposit",
    "fixed_deposits": "fixed_deposit",
    "term_deposit": "fixed_deposit", "term_deposits": "fixed_deposit",
    "au_savings_account": "savings_account",
    "personal_loan": "au_personal_loan",
    "personal_loans": "au_personal_loan",
    "au_personal_loans": "au_personal_loan",
    "green_fd": "green_fixed_deposit",
    "green_fixed_deposits": "green_fixed_deposit",
    "bulk_fd": "bulk_fixed_deposit",
    "bulk_fixed_deposits": "bulk_fixed_deposit",
}


def canonical_product(product):
    key = normalize_product(product).replace(" ", "_")
    return PRODUCT_CANONICAL_MAP.get(key, key)


def is_generic_savings_question(question):
    """Only generic savings queries; never override an explicitly named variant."""
    q = normalize_product(question.split("\nProduct specified by user:")[0])
    if not re.search(r"\bsavings?\s+accounts?\b", q):
        return False
    specific = ("regular", "digital", "aarambh", "smart saver", "bsbda", "basic",
                "value", "ivy", "royale", "kids", "kisan", "shakti", "swadesh",
                "senior citizen", "platinum", "abhi", "institution", "women")
    return not any(re.search(r"(?<!\w)" + re.escape(x) + r"(?!\w)", q) for x in specific)


def savings_overview_urls(records, indexed_urls=None):
    """Grounded in metadata, not a fabricated account variant."""
    available = {normalize_url(u) for u in indexed_urls} if indexed_urls is not None else None
    return [r["url"] for r in records
            if urlsplit(r["url"]).path.rstrip("/").lower() == "/personal-banking/savings-account"
            and (available is None or normalize_url(r["url"]) in available)]


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


# Only verified/approved aliases should be added here. The spelling "Royal-E"
# is treated as AU Royale Savings Account for this deployment; verify with the bank.
PRODUCT_ALIASES = {
    "royal e account": "au_royale_savings_account",
    "royal e": "au_royale_savings_account",
    # Royale without an account type is ambiguous: never force savings.

    "ivy account": "au_ivy_savings_account",
    "au ivy account": "au_ivy_savings_account",
    "bulk fd": "bulk_fixed_deposit",
    "fd": "fixed_deposit",
    "fds": "fixed_deposit",
    "green fd": "green_fixed_deposit",
    "green fixed deposit": "green_fixed_deposit",
    "fixed deposits": "fixed_deposit",
    "term deposits": "fixed_deposit",
}


def _find_products(question, products):
    """Detect one or more products, treating canonical aliases as one product."""
    q = normalize_product(question)
    available = set(canonical_product(p) for p in products)
    found, spans = [], []

    def add_matches(alias, target):
        alias = normalize_product(alias)
        target = canonical_product(target)
        if not alias or target not in available:
            return
        for match in re.finditer(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", q):
            b, e = match.span()
            if not any(b < old_e and e > old_b for old_b, old_e in spans):
                spans.append((b, e))
                found.append(target)

    # Longer aliases must take precedence (e.g. Green FD over FD).
    for alias, target in sorted(PRODUCT_ALIASES.items(),
                                key=lambda kv: -len(normalize_product(kv[0]))):
        add_matches(alias, target)

    for product in sorted(products, key=lambda p: -len(normalize_product(p))):
        add_matches(product, product)

    # Short forms only when they have one unique canonical owner.
    alias_owners = {}
    for product in products:
        for alias in _aliases(product) - {normalize_product(product)}:
            alias_owners.setdefault(alias, set()).add(canonical_product(product))
    for alias in sorted(alias_owners, key=len, reverse=True):
        owners = alias_owners[alias]
        if len(alias) >= 3 and len(owners) == 1:
            add_matches(alias, next(iter(owners)))
    return list(dict.fromkeys(found))


def _recommendation_intent(question):
    q = normalize_product(question)
    return bool(re.search(r"\b(best|recommend|suggest|suitable|which|looking for|need|want|zero balance|lifetime free|no annual fee)\b", q))


def _recommendation_family(question):
    q = normalize_product(question)
    if re.search(r"\b(credit card|credit cards|cards?)\b", q):
        return "card"
    if re.search(r"\b(savings account|zero balance|student account)\b", q):
        return "account"
    return None


def route_question(question, records, conversation_state=None):
    state = dict(conversation_state or {})
    products = distinct_products(records)
    detected = _find_products(question, products)
    previous_raw = state.get("products") or ([state["product"]] if state.get("product") else [])
    previous = list(dict.fromkeys(canonical_product(p) for p in previous_raw))
    products = list(dict.fromkeys(canonical_product(p) for p in products))
    pending = state.get("pending_question")
    family = _requested_family(question)
    q = normalize_product(question)
    recommendation = _recommendation_intent(question)
    rec_family = _recommendation_family(question)
    print("[ROUTER DEBUG] Detected:", detected, "Previous:", previous, "Pending:", pending,
          "Recommendation:", recommendation, "Family:", rec_family)

    # Generic product labels should not block discovery, nor be treated as
    # a second product when the user asks for a recommendation.
    generic = {"loan", "account", "deposit", "card", "credit_card", "savings_account"}
    if recommendation and rec_family and not any(p not in generic and _product_family(p) == rec_family for p in detected):
        state.update(product=None, products=[], pending_question=None)
        return {"status": "recommend", "intent": "recommendation", "family": rec_family,
                "products": [], "question": question, "state": state}

    if detected:
        # A clarification reply must not replace the actual pending question.
        short_reply = len(q.split()) <= 5 and not re.search(
            r"\b(what|how|why|compare|difference|differ|rate|interest|eligible|benefit)\b", q)
        resolved = (pending + "\nProducts: " + ", ".join(detected)) if pending and short_reply else question
        state.update(product=detected[0] if len(detected)==1 else None,
                     products=detected, pending_question=None)
        return {"status": "ready", "product": detected[0], "products": detected,
                "intent": "comparison" if len(detected)>1 else "single",
                "question": resolved, "state": state}

    # A short alias clarification, including FD, retains the ORIGINAL question.
    if pending and q in {"fd", "fds", "fixed deposit", "fixed deposits"} and "fixed_deposit" in products:
        state.update(product="fixed_deposit", products=["fixed_deposit"], pending_question=None)
        return {"status":"ready", "product":"fixed_deposit", "products":["fixed_deposit"],
                "intent":"single", "question":pending + "\nProduct: fixed_deposit", "state":state}

    # If a user is continuing a known product conversation, keep it.
    # Never turn an interest-rate follow-up into a new clarification.
    if previous and len(previous)==1 and not pending and not family and _looks_like_followup(question):
        state.update(product=previous[0], products=previous)
        return {"status":"ready", "product":previous[0], "products":previous,
                "intent":"single", "question":question, "state":state}

    # Generic family question: clarify only when no relevant product context.
    if family:
        options=[p for p in products if _product_family(p)==family and normalize_product(p) not in
                 {"loan","loans","account","accounts","deposit","deposits","card","cards"}]
        if previous and len(previous)==1 and previous[0] in options and not pending:
            state.update(product=previous[0], products=previous, pending_question=None)
            return {"status":"ready", "product":previous[0], "products":previous,
                    "intent":"single", "question":question, "state":state}
        state.update(product=None, products=[], pending_question=pending or question)
        return {"status":"clarify", "response":f"Which type of {family} are you interested in? For example: {_family_examples(options)}.", "state":state}

    # A new substantive question supersedes an unresolved clarification.
    if pending and (len(q.split())>5 or re.search(r"\b(what|how|why|compare|difference|differ)\b",q)):
        pending=None;state["pending_question"]=None
    if pending:
        return {"status":"clarify", "response":"Please specify the product name so I can answer accurately.", "state":state}
    if previous and _looks_like_followup(question):
        state.update(products=previous,product=previous[0] if len(previous)==1 else None)
        return {"status":"ready", "product":previous[0], "products":previous,
                "intent":"comparison" if len(previous)>1 else "single",
                "question":question, "state":state}
    if is_product_clarification(question):
        state.update(product=None,products=[],pending_question=question)
        return {"status":"clarify","response":"Which product are you referring to?","state":state}
    return {"status":"unsupported","response":"Sorry, I don't have information about that product in the available knowledge base.","state":state}


def recommendation_candidates(question, family, records, indexed_urls=None, limit=24):
    """Discover real indexed product pages, not just broad explainer blogs."""
    available = {normalize_url(u) for u in indexed_urls} if indexed_urls is not None else None
    q = normalize_product(question)
    is_travel = bool(re.search(r"\b(travel|foreign|international|forex|abroad)\b",q))
    is_free = bool(re.search(r"\b(lifetime free|no annual fee|zero annual fee|free credit card)\b",q))
    is_zero = bool(re.search(r"\b(zero balance|no minimum balance|student)\b",q))
    results=[]
    for item in records:
        if available is not None and normalize_url(item['url']) not in available:continue
        url=item['url'].lower(); summary=normalize_product(item['summary']); names=' '.join(item['product']).lower()
        if family=='card' and not ('credit_card' in names or 'credit-card' in url):continue
        if family=='account' and not ('savings_account' in names or 'savings-account' in url):continue
        score=0
        if family=='card' and ('/credit-cards/' in url or '/credit-card/' in url):score+=3
        if family=='account' and ('/savings-account/' in url or '/accounts/' in url):score+=3
        if '/blogs/' in url:score-=5
        if is_travel:
            for token in ('ixigo','forex','travel','international','lounge','cross currency','zero forex'):
                if token in (url+' '+summary+' '+names):score+=4 if token=='ixigo' else 1
        if is_free:
            for token in ('lifetime free','no annual fee','zero annual fee','lifetime_free'):
                if token in (summary+' '+names):score+=5
        if is_zero:
            for token in ('zero balance','no minimum balance','student','zero_balance'):
                if token in (summary+' '+names):score+=5
        if score>0:results.append((score,item))
    results.sort(key=lambda x:(-x[0],x[1]['url']))
    return [item for _,item in results[:limit]]


def rank_metadata_urls(question, product, records, embedding_model,
                       limit=12, indexed_urls=None):
    available = (
        {normalize_url(url) for url in indexed_urls}
        if indexed_urls is not None else None
    )
    candidates = [
        item for item in records
        if any(canonical_product(p) == canonical_product(product)
               for p in item["product"])
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

    # Stronger preference for authoritative product pages when the question
    # asks for an exact opening amount, eligibility requirement, or rate.
    exact_requirement = bool(re.search(r"opening|open\b|deposit|minimum|balance|require|eligib", query, re.I))
    # Prefer core product/rate pages over incidental mentions in blogs.
    for i, item in enumerate(candidates):
        path = urlsplit(item["url"]).path.lower()
        if item["category"] == "main":
            scores[i] += 0.16 if exact_requirement else 0.08
        if exact_requirement and "/blogs/" in path:
            scores[i] -= 0.08
        if "interest" in query.lower() and "interest-rate" in path:
            scores[i] += 0.10

    positions = np.argsort(-scores)[:limit]

    return [
        {**candidates[int(i)], "score": float(scores[int(i)])}
        for i in positions
    ]


def _is_specific_product_source(item, product):
    """Reject incidental mentions for branded/variant products.

    Only a page whose URL slug or summary explicitly names the full product
    can be selected for a specific variant. This is intentionally conservative.
    """
    name = normalize_product(product)
    # Generic services (fund_transfer, fixed_deposit) need broader help pages.
    tokens = [t for t in name.split() if t not in {"au", "bank", "account", "accounts"}]
    distinctive = [t for t in tokens if t in {"royale", "ivy", "aarambh", "shakti", "swadesh", "green", "bulk"}]
    if not distinctive:
        return True
    path = normalize_product(urlsplit(item["url"]).path)
    summary = normalize_product(item.get("summary", ""))
    # A generic article that merely refers to the branded product in its body
    # is not a dedicated source; require branding in the URL path or title-like
    # beginning of the summary.
    if not all(t in path or t in summary[:140] for t in distinctive):
        return False
    if "royale" in distinctive and "business" in tokens:
        if not ("royale business" in path or "royale business" in summary[:140]):
            return False
    return True


def _explicitly_named_products(question, available_products):
    """Find product IDs mentioned in the CURRENT question, not chat history."""
    q = normalize_product(question)
    found = _find_products(question, available_products)
    # Match specific full product names before generic aliases such as "loan".
    # Example: "processing fee on personal loan" -> au_personal_loan.
    specific_matches = []
    for product in available_products:
        canonical = canonical_product(product)
        if canonical in {"loan", "account", "deposit", "card"}:
            continue
        full_name = normalize_product(product)
        phrases = [full_name]
        if full_name.startswith("au "):
            phrases.append(full_name[3:])
        if any(len(phrase.split()) >= 2 and _mentioned(q, phrase)
               for phrase in phrases):
            specific_matches.append(canonical)
    if specific_matches:
        found = specific_matches + [p for p in found
                                     if canonical_product(p) not in {"loan", "account", "deposit", "card"}]
    # Handle a generic service whose canonical ID may not appear as an alias.
    if re.search(r"\bfund\s+transfers?\b", q):
        for p in available_products:
            if canonical_product(p) == "fund_transfer" and p not in found:
                found.append(p)
    return list(dict.fromkeys(canonical_product(p) for p in found))


def _is_contextual_followup(question, pending_question=None):
    q = normalize_product(question)
    if pending_question and len(q.split()) <= 6 and not re.match(
        r"^(tell me|explain|describe|what is|what are|give me|show me)\b", q
    ):
        return True
    return bool(re.search(r"\b(it|its|those|these|that|their|same|above)\b", q)
                or re.match(r"^(and|also|what about|how about|tell me more)\b", q))


def select_urls_with_llm(question, product, candidates, chat_model,
                         max_urls=3):
    if not candidates:
        return []
    candidates = [c for c in candidates if _is_specific_product_source(c, product)]
    if not candidates:
        return []

    # Reuse the existing model handler; validate its output strictly.
    from inference import run_chat_model

    shortlist = [
        {
            "id": i,
            "url": c["url"],
            "category": c["category"],
            "summary": c["summary"][:500]
        }
        for i, c in enumerate(candidates)
    ]

    instruction = """
Select the best AU Bank source URLs for answering the user question.
Only select AU Bank product/service pages relevant to the question.
Prefer primary AU Bank product pages and official terms over generic blog articles when both are relevant.
Never select a page whose principal subject is another bank.

Return ONLY JSON:
{"ids": [0, 1]}

Rules:
- Select up to 3 candidate IDs.
- Use only IDs in the supplied list.
- Prefer the authoritative page directly answering the question.
- For an opening deposit question, distinguish initial funding from average/minimum balance.
- Do not choose a blog merely because it mentions the account if a more specific official product page exists.
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


# Metadata-driven LLM routing. No fixed product alias/category table is used
# for deciding intent, scope, or product. The selected chat model makes the decision.
def route_question_llm(question, records, conversation_state=None, chat_model=None):
    from inference import run_chat_model
    state = dict(conversation_state or {})
    if not chat_model:
        raise ValueError("A selected chat model is required for metadata routing")

    # Keep classification small: product identifiers are sufficient for routing.
    # URL summaries are used later during URL ranking, not in this classifier.
    catalog = {str(product).strip(): [] for record in records
               for product in record.get("product", []) if str(product).strip()}
    recent_user_turns = [m.get("content", "") for m in state.get("history", [])
                         if m.get("role") == "user"][-2:]
    context = json.dumps({
        "available_product_ids": sorted(catalog),
        "previous_products": state.get("products", []),
        "recent_user_questions": recent_user_turns,
        "pending_question": state.get("pending_question"),
        "last_banking_question": state.get("last_banking_question")
    }, ensure_ascii=False)
    royale_options = sorted({p for p in catalog if "royale" in normalize_product(p)})
    rules = """You classify user queries for an AU Bank product/service chatbot.
Return ONLY a JSON object, no prose or code fences:
{"scope":"in_scope|out_of_scope|uncertain","intent":"single|comparison|recommendation|general|clarify", "products":[], "family":null, "explicit_product":false, "clarification":""}
- Decide scope by meaning, not by whether a product identifier is explicitly mentioned.
- In scope: AU Bank products/services, banking queries that can reasonably be answered
  from the provided AU Bank knowledge base, and contextual follow-ups.
- Out of scope: clearly unrelated topics such as sports, movies, recipes, coding, etc.
- For out_of_scope set intent=general, products=[], and never ask for a product.
- The product catalog is authoritative. Select only exact IDs from it; never invent IDs.
- Understand synonyms, plural/singular, misspellings, Hindi, Hinglish, and other Indian languages.
- Select multiple products ONLY for an explicit comparison in the CURRENT question.
- For intent=clarify, return products=[] and a brief clarification; NEVER list every candidate ID.
- Reuse previous products ONLY for genuine follow-ups, never for a standalone new topic.
- A new query about fund transfer must not inherit a previously discussed account.
- For product recommendations set intent=recommendation and family to card/account/loan/deposit
  when identifiable, even if no specific product was named.
- An attribute or audience (e.g. senior citizens, students, women, salaried people)
  is NOT itself a selected product. A question asking an interest rate, fee, eligibility,
  tenure, or other product-specific value for an audience, without identifying the
  banking product, MUST use intent=clarify, products=[], explicit_product=false.
- Do NOT infer a savings account, FD, loan, or other product solely because one
  catalog product contains that audience term. The presence of a catalog match
  does not establish the user's intended product.
- A generic Savings Account request is not an explicit Regular Savings Account request.
- Set explicit_product=true ONLY if the user actually identifies a product or
  clearly refers to an already established product in a genuine follow-up.
- If a product-specific question has multiple plausible products and no product
  is established, ask a brief choice-based clarification. Use actual relevant
  product names from the catalog. Do not guess.
- Broad questions about AU Bank's general services can use intent=general,
  products=[], and need no clarification.
- Do not ask for tenure when the user requests the full rate schedule; tenure
  clarification is needed only when a single tenure-specific calculation requires it.
- When pending_question exists, a short product reply resolves that pending question.
  Classify the reply using the catalog and answer the combined original request.
- Respond in the user's language when generating clarification text.
- Clarifications must mention AU Bank products only, using names present in the catalog.
- Never suggest or mention products of any other bank.
"""
    rules += ("\nRoyale products present in metadata: " + json.dumps(royale_options)
              + "\nFor a generic Royale account query, offer all distinct relevant account variants "
                "from this list, not just Savings and Current. Do not invent variants. "
                "For clarification replies, preserve the pending original question and its attribute "
                "(for example DAILY TRANSFER LIMIT), rather than answering how to transfer.")
    data = None
    for attempt in range(2):
        raw = run_chat_model(
            chat_model, context, question, [],
            custom_instructions=rules + ("\nIMPORTANT: Output the complete JSON object only."
                                         if attempt else ""),
            max_output_tokens=1200 if attempt else 800, temperature=0
        )
        try:
            match = re.search(r"\{[\s\S]*\}", str(raw or ""))
            if match:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, dict) and candidate.get("scope") in {
                    "in_scope", "out_of_scope", "uncertain"
                } and candidate.get("intent") in {
                    "single", "comparison", "recommendation", "general", "clarify"
                }:
                    data = candidate
                    break
        except (ValueError, TypeError):
            pass
        print(f"[ROUTER] Invalid JSON on attempt {attempt + 1}; retrying if possible")
    if data is None:
        # Avoid an HTTP 500 when classification output is truncated.
        # Do not invent a product selection.
        state["pending_question"] = question
        return {"status": "clarify", "response":
                "I couldn't process that fully. Could you please try your question again?",
                "state": state}
    scope = data.get("scope")
    intent = data.get("intent")
    if scope not in {"in_scope", "out_of_scope", "uncertain"}:
        raise ValueError("LLM router returned invalid scope")
    if intent not in {"single", "comparison", "recommendation", "general", "clarify"}:
        raise ValueError("LLM router returned invalid intent")
    if scope == "out_of_scope":
        state.update(pending_question=None)
        return {"status":"out_of_scope", "state":state}
    raw_products = data.get("products", [])
    if not isinstance(raw_products, list):
        raw_products = []
    # Resolve IDs against metadata; canonical variants may be equivalent but
    # are not used to classify the user query.
    selected = []
    for item in raw_products:
        if not isinstance(item, str):
            continue
        if item in catalog:
            chosen = item
        else:
            options = [p for p in catalog if canonical_product(p) == canonical_product(item)]
            chosen = options[0] if len(options) == 1 else None
        if chosen and chosen not in selected:
            selected.append(chosen)
    # The model's exact catalog product ID is authoritative. A generic
    # family such as "loan" must not replace "au_personal_loan".
    # Also handle a model response that includes both the generic and
    # specific IDs: keep the specific ID alone for single-product queries.
    exact_specific = [item for item in raw_products
                      if isinstance(item, str) and item in catalog
                      and item not in {"loan", "account", "deposit", "card"}]
    if exact_specific and intent != "comparison":
        selected = list(dict.fromkeys(exact_specific))[:1]

    # A generic family label is never a substitute for an explicit, specific
    # product ID returned by the classifier. In particular, do not turn
    # au_personal_loan into loan merely because both belong to the loan family.
    specific = [p for p in selected if p not in {"loan", "account", "deposit", "card"}]
    if specific:
        selected = specific

    # If the model names a generic product despite an explicit specific
    # product in the question, prefer the matching catalog product.
    explicit_specific = [p for p in _explicitly_named_products(question, list(catalog))
                         if p not in {"loan", "account", "deposit", "card"}]
    if explicit_specific and not any(canonical_product(p) in explicit_specific for p in selected):
        selected = [p for p in catalog if canonical_product(p) in explicit_specific]

    # Cross-product daily transfer limits are answered from indexed AU Bank
    # transfer-limit pages by main.py; they do not require a method clarification.
    daily_transfer = bool(re.search(r"\b(daily|per\s*day|day)\b", question, re.I)
                          and re.search(r"\btransfers?\b", question, re.I)
                          and re.search(r"\blimits?\b", question, re.I))
    if daily_transfer:
        state.update(products=["fund_transfer"], product="fund_transfer",
                     pending_question=None, last_banking_question=question)
        return {"status": "ready", "intent": "single", "products": ["fund_transfer"],
                "product": "fund_transfer", "question": question, "state": state}

    # Current-turn product mentions take precedence over stale conversation
    # state and over extra IDs hallucinated by the classifier. Explicit
    # comparisons retain all explicitly mentioned products.
    explicit_now = _explicitly_named_products(question, list(catalog))
    if explicit_now and not _is_contextual_followup(question, state.get("pending_question")):
        selected = [p for p in selected if canonical_product(p) in explicit_now]
        if not selected:
            selected = [p for p in catalog if canonical_product(p) in explicit_now]
    elif not _is_contextual_followup(question, state.get("pending_question")):
        # Never reuse an old product on a fresh query with no explicit product.
        old = {canonical_product(p) for p in state.get("products", [])}
        selected = [p for p in selected if canonical_product(p) not in old]

    # A fresh, generic savings question must not inherit a specific variant.
    # The overview is a verified metadata record and is answered via FAISS.
    if is_generic_savings_question(question) and not state.get("pending_question"):
        overview = savings_overview_urls(records)
        if overview:
            state.update(products=["savings_account"], product="savings_account",
                         pending_question=None, last_banking_question=question)
            return {"status": "ready", "intent": "single", "products": ["savings_account"],
                    "product": "savings_account", "question": question, "state": state}
    # A short NEW product question is not automatically a follow-up.
    # Only an outstanding clarification can inherit its original question.
    pending_question = state.get("pending_question")
    normalized_question = normalize_product(question)
    fresh_request = bool(re.match(
        r"^(tell me|explain|describe|what is|what are|give me|show me|"
        r"i want to know|information about|details about)\b", normalized_question))
    product_only = (bool(selected) and len(normalized_question.split()) <= 6
                    and not is_product_clarification(question) and not fresh_request)
    # A clarification response should be a short product or option name,
    # not an independent sentence asking for new information.
    resolves_pending = bool(pending_question and product_only)
    # Clarification must win over a speculative product ID. The model, not a
    # hardcoded list of products/keywords, decides whether a product was specified.
    if intent == "clarify" and scope == "in_scope":
        if re.search(r"\broyale\b", normalize_product(question)) and royale_options:
            labels = ["Savings", "Current", "Salary", "Trade", "World", "Business"]
            available = {normalize_product(p) for p in royale_options}
            labels = [label for label in labels if any(label.lower() in name for name in available)]
            if len(labels) > 1:
                data["clarification"] = ("Which AU Royale account would you like to know about: "
                                         + ", ".join(labels)
                                         + "? Royale Business also has profession-specific variants.")
        message = str(data.get("clarification") or "Which AU Bank product or service do you mean?").strip()
        state.update(products=[], product=None, pending_question=question,
                     last_banking_question=question)
        return {"status":"clarify", "response":message, "state":state}
    if resolves_pending:
        resolved_question = pending_question + "\nProduct specified by user: " + question
    else:
        resolved_question = question
    if intent == "recommendation" and pending_question and selected:
        intent = "single" if len(selected) == 1 else "comparison"
    if intent == "recommendation":
        family = data.get("family")
        if family in {"card", "account", "loan", "deposit"}:
            state.update(products=selected, product=selected[0] if len(selected)==1 else None,
                         pending_question=None)
            return {"status":"recommend", "intent":"recommendation", "family":family,
                    "products":selected, "question":resolved_question, "state":state}
    if selected:
        state["last_banking_question"] = resolved_question
        state.update(products=selected, product=selected[0] if len(selected)==1 else None,
                     pending_question=None)
        return {"status":"ready", "intent":"comparison" if len(selected)>1 else "single",
                "products":selected, "product":selected[0], "question":resolved_question, "state":state}
    if intent == "clarify" and scope == "in_scope":
        message = str(data.get("clarification") or "Could you clarify which AU Bank service you mean?").strip()
        state["pending_question"] = question
        return {"status":"clarify", "response":message, "state":state}
    state.update(products=[], product=None, pending_question=None)
    return {"status":"general", "intent":"general", "products":[],
            "question":question, "state":state}
