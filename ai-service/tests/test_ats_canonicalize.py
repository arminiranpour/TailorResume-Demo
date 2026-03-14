import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app.ats.canonicalize import (  # noqa: E402
    canonicalize_term,
    canonicalize_terms,
    extract_ngrams,
    normalize_phrase,
    normalize_text,
    tokenize_text,
)


def test_normalize_text_and_tokenize_text_are_deterministic():
    text = "  NodeJS, ReactJS, RESTful APIs, CI/CD, C Sharp, dot net  "
    assert normalize_text(text) == "nodejs reactjs restful apis ci/cd c sharp dot net"
    assert tokenize_text(text, drop_stopwords=True) == [
        "node.js",
        "react",
        "restful",
        "api",
        "ci/cd",
        "c",
        "sharp",
        "dot",
        "net",
    ]


def test_canonicalize_term_handles_safe_phrase_mappings():
    assert canonicalize_term("JS") == "javascript"
    assert canonicalize_term("TS") == "typescript"
    assert canonicalize_term("NodeJS") == "node.js"
    assert canonicalize_term("RESTful APIs") == "rest api"
    assert canonicalize_term("C Sharp") == "c#"
    assert canonicalize_term("dot net") == ".net"
    assert canonicalize_term("CI CD") == "ci/cd"


def test_canonicalize_terms_and_phrase_helpers_preserve_order():
    assert canonicalize_terms(["ReactJS", "Postgres", "Machine Learning"]) == [
        "react",
        "postgresql",
        "machine learning",
    ]
    assert normalize_phrase(" Machine   Learning ") == "machine learning"
    assert extract_ngrams(["machine", "learning", "engineer"], min_n=2, max_n=3) == [
        "machine learning",
        "learning engineer",
        "machine learning engineer",
    ]
