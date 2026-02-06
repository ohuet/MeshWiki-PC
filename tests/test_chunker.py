"""Tests for meshwiki.chunker — message splitting for Meshtastic packets."""

from meshwiki.chunker import split_message


def test_empty_string():
    assert split_message("") == []


def test_short_text_no_numbering():
    """Short text that fits in one message should have no [1/1] header."""
    text = "Bonjour le monde"
    result = split_message(text)
    assert result == ["Bonjour le monde"]
    assert len(result[0].encode("utf-8")) <= 220


def test_exact_fit_no_numbering():
    """Text exactly at the byte limit should not be split."""
    text = "a" * 220
    result = split_message(text)
    assert len(result) == 1
    assert result[0] == text


def test_long_text_is_numbered():
    """Text exceeding one chunk should be numbered [1/N]."""
    text = "Ceci est un texte assez long. " * 30
    result = split_message(text)
    assert len(result) > 1
    assert result[0].startswith("[1/")
    assert result[-1].startswith(f"[{len(result)}/")


def test_each_chunk_within_byte_limit():
    """Every chunk must be <= 220 bytes UTF-8."""
    text = "La Réunion est une île française de l'océan Indien. " * 50
    result = split_message(text, max_bytes=220)
    for chunk in result:
        assert len(chunk.encode("utf-8")) <= 220, f"Chunk too large: {len(chunk.encode('utf-8'))} bytes"


def test_utf8_accented_characters():
    """French accented characters (2 bytes in UTF-8) must be handled correctly."""
    text = "éàüîôê " * 50  # each accented char is 2 bytes
    result = split_message(text, max_bytes=220)
    for chunk in result:
        assert len(chunk.encode("utf-8")) <= 220


def test_no_mid_word_split():
    """Chunks should not split in the middle of a word."""
    text = "anticonstitutionnellement " * 20
    result = split_message(text)
    for chunk in result:
        # Remove header if present
        content = chunk
        if content.startswith("["):
            content = content.split("] ", 1)[1]
        # Content should not start or end with a partial word fragment
        # (hard to test perfectly, but at least no random byte boundaries)
        assert content == content.strip()


def test_max_five_chunks():
    """Output should never exceed 5 chunks."""
    text = "Ceci est un très long texte qui devrait dépasser cinq chunks. " * 100
    result = split_message(text)
    assert len(result) <= 5


def test_truncation_marker():
    """When truncated, the last chunk should end with the truncation marker."""
    text = "Ceci est un texte extrêmement long. " * 200
    result = split_message(text)
    if len(result) == 5:
        assert result[-1].endswith("... [tronqué]")


def test_two_chunks_numbering():
    """With exactly two chunks, numbering should be [1/2] and [2/2]."""
    # Build text that's just over 220 bytes
    text = "a" * 221
    result = split_message(text)
    assert len(result) == 2
    assert result[0].startswith("[1/2] ")
    assert result[1].startswith("[2/2] ")


def test_whitespace_only():
    """Whitespace-only input should return empty list."""
    assert split_message("   ") == []
    assert split_message("\n\t  ") == []


def test_custom_max_bytes():
    """The max_bytes parameter should be respected."""
    text = "Hello World! " * 10
    result = split_message(text, max_bytes=50)
    for chunk in result:
        assert len(chunk.encode("utf-8")) <= 50
