"""Split long messages into Meshtastic-compatible chunks (<= 220 bytes UTF-8)."""

MAX_CHUNKS = 5
TRUNCATION_MARKER = "... [tronqué]"


def split_message(text: str, max_bytes: int = 220) -> list[str]:
    """Split text into numbered chunks that fit within max_bytes UTF-8.

    - Single-message texts are returned without numbering.
    - Multi-part messages are numbered [1/N].
    - Never splits mid-word.
    - Maximum MAX_CHUNKS chunks; excess is truncated with a marker.
    """
    text = text.strip()
    if not text:
        return []

    if len(text.encode("utf-8")) <= max_bytes:
        return [text]

    # First pass: estimate how many chunks we need to determine header size.
    # We'll do a two-pass approach: split generously, then finalize.
    chunks = _split_into_chunks(text, max_bytes, total_hint=MAX_CHUNKS)

    if len(chunks) <= 1:
        return chunks

    # If we got more than MAX_CHUNKS, truncate and redo the last chunk.
    if len(chunks) > MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
        # Replace last chunk with truncation marker
        last_chunk = chunks[-1]
        marker = TRUNCATION_MARKER
        header = f"[{MAX_CHUNKS}/{MAX_CHUNKS}] "
        available = max_bytes - len(header.encode("utf-8")) - len(marker.encode("utf-8"))
        truncated_text = _truncate_to_bytes(last_chunk, available)
        chunks[-1] = truncated_text + marker

    # Apply numbering
    total = len(chunks)
    numbered = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}/{total}] "
        numbered.append(header + chunk)

    return numbered


def _split_into_chunks(text: str, max_bytes: int, total_hint: int) -> list[str]:
    """Split text into raw chunks (without numbering) that will fit once headers are added."""
    chunks = []
    remaining = text

    while remaining:
        # Reserve space for the worst-case header like "[5/5] " (7 bytes)
        header_reserve = len(f"[{total_hint}/{total_hint}] ".encode("utf-8"))
        available = max_bytes - header_reserve

        if len(remaining.encode("utf-8")) <= available:
            chunks.append(remaining)
            break

        # Find the longest prefix that fits in available bytes
        chunk_text = _truncate_to_bytes(remaining, available)

        # Try to break on whitespace or punctuation
        break_pos = _find_break_point(chunk_text)
        if break_pos > 0:
            chunk_text = chunk_text[:break_pos]

        chunks.append(chunk_text)
        remaining = remaining[len(chunk_text):].lstrip()

        # Safety: avoid infinite loop
        if not chunk_text:
            break

    return chunks


def _truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Return the longest prefix of text that fits within max_bytes UTF-8."""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text

    # Truncate the byte string and decode safely
    truncated = encoded[:max_bytes]
    # Decode ignoring incomplete multi-byte chars at the end
    result = truncated.decode("utf-8", errors="ignore")

    # If decoding lost a character, we might have cut mid-char.
    # Verify we didn't cut a character in half by re-encoding.
    while result and len(result.encode("utf-8")) > max_bytes:
        result = result[:-1]

    return result


def _find_break_point(text: str) -> int:
    """Find the best position to break text (last space or punctuation)."""
    # Search backwards for a good break point
    for i in range(len(text) - 1, 0, -1):
        if text[i] in " \t\n":
            return i
        if text[i] in ".,;:!?)-":
            return i + 1  # Keep the punctuation in this chunk

    return 0  # No good break point found
