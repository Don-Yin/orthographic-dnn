"""read a word corpus from a text file, returning non-empty lines."""

from pathlib import Path


def read_corpus(path: Path) -> list[str]:
    """read words from a newline-delimited text file, filtering blanks."""
    return [w for w in path.read_text().split("\n") if w]
