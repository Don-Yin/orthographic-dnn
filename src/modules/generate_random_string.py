"""generate random strings and save to assets."""

import json
import random
import string

from config import PATH_RANDOM_STRINGS


def random_word(length: int) -> str:
    """generate a random lowercase string of fixed length."""
    return "".join(random.choices(string.ascii_lowercase, k=length))


if __name__ == "__main__":
    random_strings = [random_word(6) for _ in range(1000)]
    PATH_RANDOM_STRINGS.write_text(json.dumps(random_strings))
