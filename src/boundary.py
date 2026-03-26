B_PUNCT = {".", ";", "?", "!", "\n\n"}
B_LOGIC = {"therefore", "however", "so", "but", "thus", "hence", "because"}
B_STRUCT = {"Step", "Answer:", "Therefore,"}

# Pre-lowered version of logic connectives for case-insensitive matching
_B_LOGIC_LOWER = {w.lower() for w in B_LOGIC}


def is_boundary(token_text: str) -> bool:
    """Check if a decoded token text hits the semantic boundary set.

    Handles whitespace-prefixed tokens (e.g. " therefore") by stripping.
    For logic connectives, matching is case-insensitive.
    For structural markers, matching is case-sensitive.
    """
    # Double newline — check before stripping
    if "\n\n" in token_text:
        return True

    stripped = token_text.strip()
    if not stripped:
        return False

    # Punctuation — check if the token ends with boundary punctuation
    # (tokens often contain preceding text, e.g. "value.")
    if stripped[-1] in {".", ";", "?", "!"}:
        return True

    # Logic connectives (case-insensitive)
    if stripped.lower().rstrip(",.;:") in _B_LOGIC_LOWER:
        return True

    # Structural markers (case-sensitive, prefix match)
    for marker in B_STRUCT:
        if stripped.startswith(marker):
            return True

    return False


class BoundaryTracker:
    """Tracks chunk length and fires at semantic boundaries.

    Fires when:
      1. chunk_len >= k_min AND current token is a boundary, OR
      2. chunk_len >= k_max (forced boundary regardless of token)

    After firing, chunk_len resets to 0.
    """

    def __init__(self, k_min: int = 15, k_max: int = 80):
        self.k_min = k_min
        self.k_max = k_max
        self.chunk_len = 0

    def step(self, token_text: str) -> bool:
        """Process one token. Returns True if boundary fires."""
        self.chunk_len += 1

        # Forced boundary at k_max
        if self.chunk_len >= self.k_max:
            self.chunk_len = 0
            return True

        # Soft boundary: k_min met and token is boundary
        if self.chunk_len >= self.k_min and is_boundary(token_text):
            self.chunk_len = 0
            return True

        return False

    def reset(self):
        self.chunk_len = 0
