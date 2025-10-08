from typing import Dict

"""
Simple guardrail helpers for input validation
"""

import re
from typing import List, Tuple


# ===== PII DETECTION =====

def contains_pii(text: str) -> bool:
    """Check if text has PII"""
    patterns = [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}-\d{3}-\d{4}\b',  # Phone
        r'\b\d{16}\b'  # Credit card
    ]
    return any(re.search(p, text) for p in patterns)


def detect_pii_types(text: str) -> List[str]:
    """Return list of PII types found"""
    types = []
    
    if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
        types.append("ssn")
    if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        types.append("email")
    if re.search(r'\b\d{3}-\d{3}-\d{4}\b', text):
        types.append("phone")
    if re.search(r'\b\d{16}\b', text):
        types.append("credit_card")
    
    return types


def redact_pii(text: str) -> str:
    """Replace PII with [REDACTED]"""
    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', text)
    # Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED]', text)
    # Phone
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[REDACTED]', text)
    # Credit card
    text = re.sub(r'\b\d{16}\b', '[REDACTED]', text)
    
    return text


# ===== BLOCKED TOPICS =====

def is_blocked_topic(text: str) -> bool:
    """Check for harmful/blocked keywords"""
    text_lower = text.lower()
    
    blocked = [
        'hack', 'bypass', 'crack', 'exploit',
        'jailbreak', 'override', 'disable safety'
    ]
    
    return any(word in text_lower for word in blocked)


def get_blocked_reason(text: str) -> str:
    """Return reason for blocking"""
    text_lower = text.lower()
    
    if any(w in text_lower for w in ['hack', 'crack']):
        return "unauthorized_access"
    if any(w in text_lower for w in ['bypass', 'disable']):
        return "safety_bypass"
    if 'jailbreak' in text_lower:
        return "jailbreak_attempt"
    
    return "harmful_content"


# ===== QUERY VALIDATION =====

def is_valid_query(text: str) -> Tuple[bool, str]:
    """
    Check if query is valid
    Returns: (is_valid, reason)
    """
    text = text.strip()
    
    if len(text) < 10:
        return False, "too_short"
    
    if len(text) > 1000:
        return False, "too_long"
    
    if not text:
        return False, "empty"
    
    return True, "valid"


# ===== RESPONSE MESSAGES =====

def get_blocked_message(reason: str) -> str:
    """Get user-friendly message for blocked queries"""
    messages = {
        "unauthorized_access": "I cannot help with unauthorized system access. Please contact official support.",
        "safety_bypass": "I cannot provide instructions to bypass safety features.",
        "jailbreak_attempt": "I'm designed to provide helpful, safe information only.",
        "harmful_content": "I cannot assist with that request. Please ask about vehicle features or banking services."
    }
    return messages.get(reason, messages["harmful_content"])


def get_invalid_message(reason: str) -> str:
    """Get user-friendly message for invalid queries"""
    messages = {
        "too_short": "Please provide more details about what you need help with.",
        "too_long": "Please try to be more concise and focus on one main question.",
        "empty": "Please enter a question."
    }
    return messages.get(reason, "Please enter a valid question.")

# ===== OUTPUT GUARDRAILS HELPERS =====

def extract_citations(retrieved_docs: List[Dict]) -> List[str]:
    """Extract unique source URLs from retrieved documents"""
    citations = []
    seen = set()
    
    for doc in retrieved_docs:
        # Get source URL from metadata
        source = doc.get('metadata', {}).get('source', '')
        
        if source and source not in seen:
            citations.append(source)
            seen.add(source)
    
    return citations[:5]  # Limit to top 5 sources


def extract_assets(retrieved_docs: List[Dict]) -> List[str]:
    """Extract images and PDFs from document metadata"""
    assets = []
    seen = set()
    
    for doc in retrieved_docs:
        metadata = doc.get('metadata', {})
        
        # Get images - handle None case
        images = metadata.get('images') or []
        if isinstance(images, list):  # Make sure it's a list
            for img in images:
                if img and isinstance(img, str) and img not in seen:  # Check it's a string
                    assets.append(img)
                    seen.add(img)
        
        # Get PDFs - handle None case  
        pdfs = metadata.get('pdfs') or []
        if isinstance(pdfs, list):  # Make sure it's a list
            for pdf in pdfs:
                if pdf and isinstance(pdf, str) and pdf not in seen:  # Check it's a string
                    assets.append(pdf)
                    seen.add(pdf)
    
    return assets[:5]  # Limit to 5 assets


def check_answer_safety(answer: str) -> bool:
    """Check if answer contains unsafe content"""
    unsafe_patterns = [
        'bypass', 'hack', 'disable safety', 'override',
        'unauthorized', 'illegal', 'exploit'
    ]
    
    answer_lower = answer.lower()
    return not any(pattern in answer_lower for pattern in unsafe_patterns)


def detect_output_pii(text: str) -> bool:
    """Check if answer accidentally contains PII"""
    # Reuse PII detection from input
    return contains_pii(text)