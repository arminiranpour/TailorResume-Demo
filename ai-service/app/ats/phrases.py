"""Canonical term and phrase vocab for deterministic ATS extraction."""

from __future__ import annotations

CANONICAL_TERM_MAP = {
    "api": "api",
    "apis": "api",
    "ci/cd": "ci/cd",
    "javascript": "javascript",
    "js": "javascript",
    "k8s": "kubernetes",
    "node.js": "node.js",
    "nodejs": "node.js",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "react": "react",
    "reactjs": "react",
    "ts": "typescript",
    "typescript": "typescript",
}

CANONICAL_PHRASE_MAP = {
    "c sharp": "c#",
    "ci cd": "ci/cd",
    "ci/cd": "ci/cd",
    "computer science": "computer science",
    "data pipeline": "data pipeline",
    "data pipelines": "data pipeline",
    "dot net": ".net",
    "machine learning": "machine learning",
    "node js": "node.js",
    "react js": "react",
    "rest api": "rest api",
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "software engineer": "software engineer",
}

PROTECTED_PHRASES = frozenset(
    {
        ".net",
        "computer science",
        "data pipeline",
        "machine learning",
        "rest api",
        "software engineer",
    }
)

PHRASE_HEADWORDS = frozenset(
    {
        "analyst",
        "api",
        "apis",
        "architect",
        "automation",
        "cloud",
        "database",
        "developer",
        "engineer",
        "framework",
        "infrastructure",
        "learning",
        "manager",
        "microservices",
        "pipeline",
        "pipelines",
        "platform",
        "scientist",
        "security",
        "service",
        "services",
        "stack",
        "systems",
    }
)
