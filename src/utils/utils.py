from unstructured.cleaners.core import (
    replace_unicode_quotes,
    bytes_string_to_string,
    clean_bullets,
    clean_dashes,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    clean_ordered_bullets
)

import re
from underthesea import text_normalize

def clean_unicode_quotes(text):
    """Replaces unicode bullets in text with the expected character

    Example
    -------
    \x93What a lovely quote!\x94 -> “What a lovely quote!”
    """
    # NOTE(robinson) - We should probably make this something more sane like a regex
    # instead of a whole big series of replaces
    text = replace_unicode_quotes(text)
    return text

def convert_byte_string_to_string(text, encoding="utf-8"):
    """Converts a byte string to a string using the specified encoding

    >>> text = "Hello ð\x9f\x98\x80"
    >>> # The output should be "Hello 😀"
    >>> bytes_string_to_string(text, encoding="utf-8")
    We first need to convert to string, then remove unnecessary
    string if needed, for example, icon
    """
    text = bytes_string_to_string(text, encoding=encoding)
    return text

def remove_bullets(text):
    """Removes bullets from the beginning of text
    Bullets that do not appear at the beginning of the text are not removed.
    >>> # Returns "An excellent point!"
    >>> clean_bullets("● An excellent point!")

    >>> # Returns "I love Morse Code! ●●●"
    >>> clean_bullets("I love Morse Code! ●●●")
    """
    text = clean_bullets(text)
    return text

def replace_extracwhitespace(text):
    """Removes extra whitespace
    Also handles special characters such as \xa0 and newlines.
    """
    text = clean_extra_whitespace(text)
    return text

def decode_hexadecimal(text, hexacode = b'\xe2\x80\x93'):
    """
    Replaces a specified hexadecimal code with a hyphen in a given text.

    This function takes a text and a hexadecimal code (default is b'\xe2\x80\x93' which
    represents the en dash character) and replaces all occurrences of that code with a hyphen.
    We see: "(1756-1836)" but it is b'(1756\xe2\x80\x931836)' not b'(1756-1836)',
    character `-` now is not hyphen, it is en-dash, if we go through clean non-ascii
    character, en-dash will be removed, the string results in "(17561836)".

    Args:
        text (str):
            The input text in which the hexadecimal code will be replaced.
        hexacode (bytes, optional):
            The hexadecimal code to be replaced (default is b'\xe2\x80\x93'
            which represents the en dash).

    Returns:
        str: The input text with the specified hexadecimal code replaced by hyphens.

    References:
    - https://itecnote.com/tecnote/python-why-is-the-en-dash-written-as-xe2x80x93-in-python/
    """
    encoded_text = [char.encode("utf-8") for char in text]
    while hexacode in encoded_text:
        encoded_text[encoded_text.index(hexacode)] = b'-'
    decoded_text = [char.decode() for char in encoded_text]
    return "".join(decoded_text)

def clean_wiki_text(text: str) -> str:
    """
    Clean wikipedia text by removing multiple new lines, removing extremely short lines,
    adding paragraph breaks and removing empty paragraphs
    """
    # get rid of multiple new lines
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    # remove extremely short lines
    lines = text.split("\n")
    cleaned = []
    for l in lines:
        if len(l) > 30 or (l[:2] == "==" and l[-2:] == "=="):
            cleaned.append(l)
    text = "\n".join(cleaned)

    # add paragraphs (identified by wiki section title which is always in format "==Some Title==")
    text = text.replace("\n==", "\n\n\n==")

    # remove empty paragrahps
    text = re.sub(r"(==.*==\n\n\n)", "", text)

    return text

def normalize_vi_text(text: str):
    text = text_normalize(text)
    return text

def clean_text_funct(text):
    text = clean_wiki_text(text)
    text = clean_unicode_quotes(text)
    text = remove_bullets(text)
    text = replace_extracwhitespace(text)
    text = decode_hexadecimal(text)
    # text = convert_byte_string_to_string(text)
    text = normalize_vi_text(text)
    return text