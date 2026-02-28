from typing import Dict, List


def sanitize_untrusted_text(text: str) -> str:
    """
    Normalize and wrap untrusted text as data-only content.
    Does not alter semantics beyond whitespace/newline normalization.
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return (
        "<<<BEGIN_UNTRUSTED_TEXT>>>\n"
        f"{normalized}\n"
        "<<<END_UNTRUSTED_TEXT>>>"
    )


def build_llm_messages(system_rules: str, user_untrusted_text: str) -> List[Dict[str, str]]:
    """
    Build a minimal message list that keeps rules in system role and
    untrusted data in user role.
    """
    return [
        {"role": "system", "content": system_rules},
        {"role": "user", "content": sanitize_untrusted_text(user_untrusted_text)},
    ]
