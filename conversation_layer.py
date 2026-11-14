"""Presentation-only conversation policy. Never supplies financial facts."""
import re

DISCLAIMER = "Disclaimer: This response is AI-generated. Please verify banking information through official AU Bank channels."

def _norm(value):
    return re.sub(r"\s+", " ", str(value or '').strip().lower())

def contextual_opening(question, language='en'):
    """Acknowledge a clearly stated circumstance; no financial claims."""
    q = _norm(question)
    if language != 'en':
        return ''  # Let the multilingual response model express empathy naturally.
    if re.search(r"\b(birthday|bday|b-day)\b", q):
        return "Happy birthday! 🎉 Wishing you a wonderful day."
    if re.search(r"\b(wedding|getting married|marriage ceremony)\b", q) and re.search(r"\b(my|i am|i'm|our)\b", q):
        return "Congratulations on your upcoming celebration!"
    if re.search(r"\b(lost my job|laid off|job loss)\b", q):
        return "I'm sorry you're going through a difficult time."
    if re.search(r"\b(frustrated|fed up|very upset|terrible experience|useless app)\b", q):
        return "I understand how frustrating that must be. I'll do my best to help."
    return ''

def finalize_answer(answer, question, language='en', substantive=True):
    """Preserve Related Pages section and enforce a single disclaimer on banking answers."""
    if not isinstance(answer, str):
        return answer
    pieces = re.split(r'(?im)^\s*Related search results\s*:\s*$', answer, maxsplit=1)
    body = pieces[0].strip()
    if not body:
        return answer
    # Avoid double greetings when the model already acknowledges the milestone.
    opener = contextual_opening(question, language)
    if opener and not re.search(r'(?i)\b(happy birthday|congratulations|sorry you|understand how frustrating)\b', body[:220]):
        body = opener + '\n\n' + body
    if substantive and 'disclaimer:' not in body.lower():
        body += '\n\n' + DISCLAIMER
    return body + ('\n\nRelated search results:\n' + pieces[1].strip() if len(pieces) > 1 else '')


# Conversation-level controls are intentionally independent of financial retrieval.
# They prevent an insulting one-word message being misclassified as out-of-scope.
_ABUSE = re.compile(
    r"(?i)(?<!\w)(?:asshole|bastard|fuck(?:ing|er)?|motherfucker|"
    r"bsdk|bhosdike|bhosdi(?:ke)?|madarchod|mc|behenchod|bc|"
    r"chutiya|chut|gaand|gandu|saale|harami|kamina)(?!\w)"
)
_BANKING_WORDS = re.compile(
    r"(?i)\b(?:loan|personal\s+loan|pl|fd|account|bank|card|emi|upi|"
    r"pin|mpin|interest|rate|deposit|transfer|balance|kyc|payment|apply)\b"
)
_SOCIAL = re.compile(
    r"(?i)^\s*(?:hi|hii+|hey+|hello|yo|sup|what'?s\s*up|whatsup|"
    r"how(?:'s| is)\s*it\s*going|how\s*are\s*you(?:\s*doing)?|"
    r"hi[,! ]+what'?s\s*up|hey[,! ]+what'?s\s*up|"
    r"hello[,! ]+how\s*are\s*you|thanks|thank\s*you|good\s*morning|"
    r"good\s*evening|good\s*night)\s*[?!.]*\s*$"
)

def is_casual_social(question):
    q = str(question or '').strip()
    return bool(_SOCIAL.fullmatch(q)) and not _BANKING_WORDS.search(q)

def conversational_guardrail(question, language='en'):
    """Handle standalone insults consistently; don't override real banking help requests."""
    q = _norm(question)
    if not _ABUSE.search(q) or _BANKING_WORDS.search(q):
        return None
    if language == 'hi-Latn':
        return ("Main madad ke liye yahan hoon. Kripya baat-cheet mein samman banaye "
                "rakhein. Agar koi dikkat hai, bataiye—main madad karne ki koshish karunga.")
    if language == 'hi':
        return ("मैं आपकी मदद के लिए यहाँ हूँ। कृपया बातचीत सम्मानजनक रखें। "
                "अगर कोई परेशानी है, तो बताइए—मैं मदद करने की कोशिश करूँगा।")
    return ("I’m here to help, and I’m happy to hear what’s bothering you. "
            "Please keep our conversation respectful so we can work through it together. "
            "What happened?")
