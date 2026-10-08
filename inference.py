# inference.py
import os
import google.generativeai as genai
import openai
import pytz
import numpy as np
from datetime import datetime
from langchain_together import ChatTogether
from dotenv import load_dotenv
from groq import Groq
import random
import re
from urllib.parse import urlparse, unquote
from functools import lru_cache

load_dotenv()

# Get API key from environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = (OPENAI_API_KEY)

# --- Token budgeting -------------------------------------------------
# Needed because some providers (Groq's on_demand tier in particular)
# enforce a hard tokens-per-minute cap per request, not just a daily
# quota — a prompt that's merely "long" will 413 every single time, no
# matter which key you use.
#
# Deliberately NOT using tiktoken here: its encoder files are fetched
# from openaipublic.blob.core.windows.net on first use, which 403/fails
# in network-restricted deployments (firewalled servers, offline CI,
# etc.) — exactly the kind of environment this token-budget guard needs
# to keep working in. A ~4-chars/token estimate is conservative enough
# for budgeting purposes (it slightly over-counts for English text, so
# it errs on the side of truncating a bit more, never less).
def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)

def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# Groq's on_demand tier for openai/gpt-oss-120b caps requests at 8000
# tokens/minute TOTAL (prompt + completion combined). Keep a safety
# margin below that for prompt-envelope overhead (chat template, role
# tokens, etc.) so an estimate that looks "just under" doesn't still 413.
GROQ_TPM_LIMIT = 8000
GROQ_SAFETY_MARGIN = 500

# --- Prompt Builder ---
def build_rag_prompt(context, history, question, current_datetime, custom_instructions=None,
                      max_output_tokens=None, max_context_tokens=None, max_history_tokens=None):
    print("[Prompt Builder] Building prompt with:")
    print("- Context length:", len(context))
    print("- Chat history:", history)
    print("- Question:", question)
    print("- Current datetime:", current_datetime)
    print("- Custom instructions:", custom_instructions is not None)

    # ⚠️ Remove duplicate question if it's the last in history
    if history.strip().endswith(f"User: {question.strip()}"):
        history = "\n".join(history.strip().split("\n")[:-1])

    # Cap context/history to whatever budget the caller gives us (this is
    # what actually prevents the "request too large" 413s on providers
    # like Groq that enforce a strict tokens-per-minute limit — without
    # this, a long conversation or a generous top_k just grows the prompt
    # unbounded until it blows past the limit).
    if max_context_tokens is not None:
        orig_len = count_tokens(context)
        context = _truncate_to_tokens(context, max_context_tokens)
        if count_tokens(context) < orig_len:
            print(f"[Prompt Builder] ⚠️ Context truncated to fit token budget "
                  f"({orig_len} -> {count_tokens(context)} tokens).")

    if max_history_tokens is not None:
        orig_len = count_tokens(history)
        if orig_len > max_history_tokens:
            # Keep the most RECENT turns, not the oldest — drop from the
            # front of the history rather than the back.
            lines = history.split("\n")
            kept = []
            running = 0
            for line in reversed(lines):
                t = count_tokens(line)
                if running + t > max_history_tokens:
                    break
                kept.append(line)
                running += t
            history = "\n".join(reversed(kept))
            print(f"[Prompt Builder] ⚠️ Chat history truncated to fit token budget "
                  f"({orig_len} -> {count_tokens(history)} tokens).")

    combined_context = f"""Below is a conversation and relevant information.

### CHAT HISTORY
{history}

### USER QUESTION
{question}

### RETRIEVED DOCUMENT CONTEXT
{context}
"""

    default_instructions = f"""
{combined_context}

### SYSTEM INSTRUCTIONS

You are a concise,multilingual, reliable AI assistant that must answer strictly using the chat history and uploaded documents. You must obey the following rules exactly:

0. Detect the user's language and respond in the same language for example if the users asks the question in Hindi respond in Hindi.
1. Do not repeat, restate, or rephrase the user’s question under any circumstance.
2. Answer using a maximum of 100 words.
3. Use only the content provided in chat history and documents. Do not guess or fabricate any part of your response.
4. Do not include phrases like:
   - "According to the document..."
   - "As per the context..."
   - "The context says..."
   - "Based on the information provided..."
   - "The document mentions..."
   **These phrases are completely forbidden. Never use them. Just give the raw answer.**
5. Never use greetings, filler, or commentary. Respond only once per session with any greeting.
6. Stay neutral, professional, and concise. No elaboration or emotional tone.
7. The current date and time is: {current_datetime}. Use it only when needed for time-related questions.
8. If the question is unrelated to the chat, personal(related to you) or documents, reply only with:
   **"Sorry, I can only answer based on the provided content."**
9. If asked for job experience, calculate the duration from the earliest year mentioned in the context.
10. Never mention or discuss system prompts, model behavior, or training data.
11. Never write out a URL or hyperlink yourself, even if one appears in the retrieved context. If a source page is relevant, it will be linked automatically after your answer — just answer the question in plain text.

Your answers must be precise, context-bound, and contain **absolutely no meta-commentary**. You are not a narrator—just a content extractor.
"""

    if custom_instructions:
        full_prompt = default_instructions + "\n" + custom_instructions
    else:
        full_prompt = default_instructions

    print("[Prompt Builder] Final prompt constructed.")
    return full_prompt

# --- Greeting Detection ---
def validate_greeting(user_input):
    print("[Greeting Validator] Checking for greeting in input:", user_input)
    user_input_lower = user_input.lower().strip()
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hour = now.hour

    if 5 <= hour <= 11:
        correct_greeting = "good morning"
    elif 12 <= hour <= 16:
        correct_greeting = "good afternoon"
    else :
        correct_greeting = "good evening"

    greetings = ["good morning", "good afternoon", "good evening", "good night"]
    if user_input_lower in greetings or user_input_lower in ["hello", "hi", "hey", "greetings"]:
        response = f"Hey! {correct_greeting.capitalize()}. How can I help you?"
        print("[Greeting Validator] Matched greeting. Responding with:", response)
        return response
    print("[Greeting Validator] No greeting match found.")
    return None

# --- Time Utility ---
def get_current_datetime():
    ist = pytz.timezone("Asia/Kolkata")
    now_str = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
    print("[Time Utility] Current datetime:", now_str)
    return now_str

# --- Source Link Helper ---
# Answers that draw on scraped website content, or on a URL embedded inside
# an uploaded PDF/DOCX, should point back to that source. We never let the
# LLM write the URL itself (models can truncate, mangle, or invent long
# URLs) — instead we deterministically append the correct link(s) here,
# based on the metadata attached to whichever chunks were actually
# retrieved for this question.
_NO_LINK_ANSWERS = {
    "sorry, i can only answer based on the provided content.",
    "no relevant context found in the documents.",
}

# Product-aware source recommendation. No fabricated or hardcoded destination URLs.
PRODUCT_TERMS = {
    "personal loan": ("personal loan", "personal loans", "personal-loan", "personal-loans", "pl loan"),
    "fixed deposit": ("fixed deposit", "fixed deposits", "fixed-deposit", "fixed-deposits", "fd", "term deposit"),
    "savings account": ("savings account", "savings accounts", "savings-account", "savings-accounts"),
    "msme loan": ("msme", "business loan", "business-loan", "business-loans", "small business loan"),
    "home loan": ("home loan", "home-loan", "housing loan"),
    "credit card": ("credit card", "credit-card", "credit cards"),
    "car loan": ("car loan", "car-loan", "auto loan"),
    "gold loan": ("gold loan", "gold-loan"),
    "current account": ("current account", "current-account"),
    "recurring deposit": ("recurring deposit", "recurring-deposit", "rd account"),
}


def _normalized_words(value):
    return re.sub(r"[^a-z0-9]+", " ", unquote(value).lower()).strip()


def _product_for_question(question):
    q = " " + _normalized_words(question) + " "
    matches = []
    for product, aliases in PRODUCT_TERMS.items():
        for alias in aliases:
            token = " " + _normalized_words(alias) + " "
            if token in q:
                matches.append((len(token), product))
    return max(matches)[1] if matches else None


def _url_kind(url):
    path = urlparse(url).path.lower()
    return "blog" if re.search(r"/(blogs?|articles?)/", path + "/") else "main"


def _valid_source(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("https", "http") and bool(parsed.netloc) and not any(
            x in parsed.path.lower() for x in (".pdf", ".jpg", ".png", ".svg", ".zip")
        )
    except (TypeError, ValueError):
        return False


def _source_score(url, product, question, retrieved_urls):
    path = _normalized_words(urlparse(url).path)
    question_words = set(_normalized_words(question).split()) - {
        "what", "which", "where", "how", "the", "for", "are", "can", "with", "does", "and", "apply", "about", "required"
    }
    path_words = set(path.split())
    score = 2 * len(question_words & path_words)
    if url in retrieved_urls:
        score += 8
    if product:
        aliases = PRODUCT_TERMS[product]
        matched = any(_normalized_words(alias) in path for alias in aliases)
        if not matched:
            return -1000
        score += 25
        # Avoid confusing products with overlapping vocabulary.
        for other_product, other_aliases in PRODUCT_TERMS.items():
            if other_product != product and any(
                _normalized_words(a) in path for a in other_aliases if len(_normalized_words(a)) >= 7
            ):
                score -= 25
    if _url_kind(url) == "main":
        # Prefer short product landing paths over calculators, FAQs and rate subpages.
        score -= max(0, len(path_words) - 3) * 2
        if any(w in path_words for w in ("calculator", "eligibility", "faq", "apply", "charges", "interest", "rates")):
            score -= 12
        if not path:
            score -= 50
    return score


def _all_index_sources(docstore):
    # Build from indexed source metadata, not from LLM-generated text.
    sources = set()
    for doc in getattr(docstore, "_dict", {}).values():
        metadata = getattr(doc, "metadata", {}) or {}
        candidates = [metadata.get("source_url")] + list(metadata.get("source_urls") or [])
        for url in candidates:
            if isinstance(url, str) and _valid_source(url):
                sources.add(url.strip())
    return sources


def append_web_sources(answer, web_sources, question="", docstore=None, max_sources=2):
    if not answer or answer.strip().lower() in _NO_LINK_ANSWERS:
        return answer
    if answer.lower().startswith(("an error occurred", "this service is temporarily unavailable")):
        return answer
    product = _product_for_question(question)
    retrieved = set(web_sources or [])
    candidates = _all_index_sources(docstore) if docstore is not None else set()
    candidates.update(url for url in retrieved if _valid_source(url))
    # For unknown products, restrict recommendations to actual retrieved evidence.
    if not product:
        candidates &= retrieved
    chosen = {}
    for kind in ("blog", "main"):
        ranked = sorted(
            (u for u in candidates if _url_kind(u) == kind),
            key=lambda u: (-_source_score(u, product, question, retrieved), len(urlparse(u).path), u),
        )
        if ranked and _source_score(ranked[0], product, question, retrieved) > 0:
            chosen[kind] = ranked[0]
    if not chosen:
        return answer
    lines = []
    if "main" in chosen and chosen["main"] != chosen.get("blog"):
            lines.append(f"Official Product Page: {chosen['main']}")
    if "blog" in chosen:
        lines.append(f"Related Blog: {chosen['blog']}")
    return answer.strip() + "\n\nExplore more:\n" + "\n".join(lines[:max_sources])

# --- Unified Chat Model Handler ---
def run_chat_model(chat_model, context, question, chat_history, custom_instructions=None, max_output_tokens=1024, temperature=0.3):
    print(f"[Model Handler] Running chat model: {chat_model}")
    print(f"[Model Handler] max_output_tokens received: {max_output_tokens}")

    # Guard clause for missing chat_model
    if not chat_model or not isinstance(chat_model, str):
        raise ValueError("[Model Handler] ❌ 'chat_model' is None or invalid. Please provide a valid model name.")

    current_datetime = get_current_datetime()
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    chat_model_lower_for_budget = chat_model.lower() if chat_model else ""
    if chat_model_lower_for_budget in ["openai/gpt-oss-120b"]:
        # Groq on_demand tier: 8000 TPM total (prompt + completion). Leave
        # room for max_output_tokens and a safety margin, then split what's
        # left between context and history (context gets the lion's share
        # since it's usually what answers the question).
        reserved = (max_output_tokens or 1024) + GROQ_SAFETY_MARGIN
        available = max(500, GROQ_TPM_LIMIT - reserved)
        max_context_tokens = int(available * 0.75)
        max_history_tokens = available - max_context_tokens
    else:
        max_context_tokens = None
        max_history_tokens = None

    prompt = build_rag_prompt(
        context, history_context, question, current_datetime, custom_instructions,
        max_output_tokens=max_output_tokens,
        max_context_tokens=max_context_tokens,
        max_history_tokens=max_history_tokens,
    )

    try:
        chat_model_lower = chat_model.lower()

        if "gemini" in chat_model_lower:
            print("[Model Handler] Using Gemini model...")
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(
                [prompt],
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens
                },
                safety_settings={
                    "HARASSMENT": "BLOCK_NONE",
                    "HATE": "BLOCK_NONE",
                    "SEXUAL": "BLOCK_NONE",
                    "DANGEROUS": "BLOCK_NONE"
                }
            )
            print("[Gemini Response]", response)
            return response.text

        elif "gpt" in chat_model_lower and chat_model_lower not in ["openai/gpt-oss-120b", "groq/compound", "groq/compound-mini"]:
            print("[Model Handler] Using OpenAI GPT model...")
            messages = [
                {"role": "system", "content": prompt}
            ]
            response = openai.ChatCompletion.create(
                model=chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens
            )
            print("[OpenAI GPT Response]", response["choices"][0]["message"]["content"])
            return response["choices"][0]["message"]["content"]

        elif chat_model_lower in ["openai/gpt-oss-120b"]:
            print("[Model Handler] Using Groq model with randomized API key rotation...")
            groq_keys = [
                os.getenv("GROQ1"),
                os.getenv("GROQ2"),
                os.getenv("GROQ3"),
                os.getenv("GROQ4"),
                os.getenv("GROQ5")
            ]
            groq_keys = [k for k in groq_keys if k]
            # Shuffle keys before trying
            random.shuffle(groq_keys)

            def _is_tpm_error(msg: str) -> bool:
                # "Request too large ... tokens per minute (TPM)" — this is a
                # per-ORG cap on Groq, not per-key, so all GROQ1..GROQ5 keys
                # (same org) will fail identically. Rotating keys can never
                # fix this; only shrinking the request can.
                return "tokens per minute" in msg or "tpm" in msg or "request too large" in msg

            def _is_key_specific_error(msg: str) -> bool:
                return any(k in msg for k in ["invalid api key", "permission", "unauthorized", "quota", "exhausted"])

            current_prompt = prompt
            tpm_retry_done = False

            attempt_keys = list(groq_keys)
            i = 0
            while i < len(attempt_keys):
                key = attempt_keys[i]
                try:
                    client = Groq(api_key=key)
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=[
                            {"role": "system", "content": current_prompt},
                            {"role": "user", "content": question}
                        ],
                        temperature=temperature,
                        max_tokens=max_output_tokens
                    )
                    print(f"[Groq Response with key ending {key[-4:]}] {response.choices[0].message.content}")
                    return response.choices[0].message.content

                except Exception as e:
                    print(f"[Groq API Key {key[-4:]} Failed] ➤ {e}")
                    error_message = str(e).lower()

                    if _is_tpm_error(error_message):
                        if not tpm_retry_done:
                            # Same request will fail on every remaining key
                            # (shared org quota) — cut the prompt hard and
                            # retry ONCE with a much smaller context instead
                            # of burning through all 5 keys pointlessly.
                            print("[Model Handler] ⚠️ Groq TPM limit hit — "
                                  "aggressively shrinking prompt and retrying once "
                                  "instead of rotating keys.")
                            emergency_budget = max(500, GROQ_TPM_LIMIT - (max_output_tokens or 1024) - 1000)
                            current_prompt = _truncate_to_tokens(current_prompt, emergency_budget)
                            tpm_retry_done = True
                            continue  # retry with the SAME key, smaller prompt
                        else:
                            # Already shrank once and it still won't fit —
                            # further key rotation is pointless, bail out.
                            return ("This request retrieved too much context for the "
                                    "model's per-minute token limit even after "
                                    "trimming. Try a more specific question or a "
                                    "lower top_k setting.")

                    if _is_key_specific_error(error_message):
                        i += 1
                        continue

                    return f"An error occurred while generating response: {str(e)}"

            return "This service is temporarily unavailable due to exhausted API usage."


        else:
            print("[Model Handler] Using Together AI model...")
            model = ChatTogether(
                together_api_key="94d32cd3eedfd9911a6b1c281bc14d278cd0e4f3e52272b3f7cbbed13e698511",
                model=chat_model
            )
            output = model.predict(prompt, max_tokens=max_output_tokens,temperature=temperature)
            print("[Together AI Response]", output)
            print("Temperature ->>>>>>>>>>>>>>>>>", temperature)
            return output

    except Exception as e:
        print(f"[Model Handler Error] {e}")
        error_message = str(e).lower()
        if any(keyword in error_message for keyword in ["quota", "exceeded", "limit", "exhausted", "invalid api key", "permission denied", "unauthorized"]):
            return "This service is temporarily unavailable due to exhausted API usage."
        return f"An error occurred while generating response: {str(e)}"
    
# --- FAISS Inference Only ---
def inference_faiss(chat_model, question, embedding_model_global, index, docstore, index_to_docstore_id, chat_history, custom_instructions=None, max_output_tokens=1024, top_k=3, temperature=0.3):
    print("[FAISS] Performing FAISS search...")
    try:
        query_embedding = embedding_model_global.embed_query(question)
        k = top_k
        D, I = index.search(np.array([query_embedding]), k=k)
        print(f"[FAISS] Top {k} indices: {I[0]}")

        contexts = []
        # Deduped list of URLs to cite after the answer. Two sources feed this:
        #   - scraped web pages: a single URL under metadata["source_url"]
        #   - uploaded PDFs/DOCX: zero or more URLs found in the doc text,
        #     under metadata["source_urls"] (see preprocess.py _extract_urls)
        web_sources = []
        for faiss_idx in I[0]:
            if faiss_idx != -1:
                docstore_id = index_to_docstore_id.get(faiss_idx)
                if docstore_id:
                    doc = docstore.search(docstore_id)
                    if hasattr(doc, "page_content"):
                        contexts.append(doc.page_content)
                        metadata = getattr(doc, "metadata", None) or {}

                        if metadata.get("source_type") == "web":
                            url = metadata.get("source_url")
                            if url and url not in web_sources:
                                web_sources.append(url)

                        # NEW: uploaded-document chunks (PDF/DOCX) can carry
                        # their own embedded URLs — cite those the same way.
                        for url in metadata.get("source_urls") or []:
                            if url and url not in web_sources:
                                web_sources.append(url)

        if not contexts:
            print("[FAISS] No documents found in retrieved indices.")
            return "No relevant context found in the documents."
        
        print("[FAISS] Retrieved documents:")
        for doc in contexts:
            print(" -", doc[:200], "...")  # Truncate to avoid log flooding
        print(f"[FAISS] Web sources among retrieved chunks: {web_sources}")

        context = "\n\n---\n\n".join(contexts)
        answer = run_chat_model(chat_model, context, question, chat_history, custom_instructions, max_output_tokens=max_output_tokens, temperature=temperature)
        return append_web_sources(answer, web_sources, question=question, docstore=docstore)
    except Exception as e:
        print(f"[FAISS ERROR] {str(e)}")
        return "An error occurred while processing your question."
    
# --- Simplified function for Optuna tuning ---
def get_response(question, top_k=3, temperature=0.3, max_output_tokens=1024, custom_docs=None):
    print("[get_response] Called from Optuna tuning")
    if not custom_docs:
        return "❌ No documents provided."

    context = "\n\n---\n\n".join([doc.page_content for doc in custom_docs if hasattr(doc, "page_content")])
    return run_chat_model(
        chat_model="llama3-8b-8192", 
        context=context,
        question=question,
        chat_history=[],
        custom_instructions=None,
        max_output_tokens=max_output_tokens
    )


# --- Dispatcher (Only FAISS retained) ---
def inference(vectordb_name, chat_model, question, embedding_model_global, chat_history, custom_instructions=None, faiss_index_dir=None, max_output_tokens=1024, top_k=8, temperature=0.3):
    print(f"[Dispatcher] Routing to {vectordb_name} inference...")
    print(f" - Chat model: {chat_model}")
    print(f" - Question: {question}")
    print(f" - Custom instructions provided: {custom_instructions is not None}")

    custom_greeting_response = validate_greeting(question)
    if custom_greeting_response:
        return custom_greeting_response

    if vectordb_name == "FAISS":
        from langchain_community.vectorstores import FAISS

        if faiss_index_dir is None:
            print("[Dispatcher] ❌ FAISS index directory not provided.")
            return "❌ FAISS index directory not provided."

        faiss_index_path = os.path.join(faiss_index_dir, "index.faiss")
        if os.path.exists(faiss_index_path):
            import faiss
            import pickle

            index = faiss.read_index(faiss_index_path)
            with open(os.path.join(faiss_index_dir, "index.pkl"), "rb") as f:
                store_data = pickle.load(f)

            faiss_store = FAISS(
                embedding_function=embedding_model_global,
                index=index,
                docstore=store_data["docstore"],
                index_to_docstore_id=store_data["index_to_docstore_id"]
            )

            return inference_faiss(
                chat_model, question, embedding_model_global,
                faiss_store.index, faiss_store.docstore, faiss_store.index_to_docstore_id,
                chat_history, custom_instructions , max_output_tokens=max_output_tokens,top_k=top_k,temperature=temperature
            )
        else:
            return f"❌ FAISS index file not found at {faiss_index_path}. Please run preprocessing first."
    else:
        print("[Dispatcher] ❌ Invalid vector DB:", vectordb_name)
        return "❌ Invalid vector database selection! Only FAISS is supported."