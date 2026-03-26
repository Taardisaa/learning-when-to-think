from src.boundary import is_boundary, BoundaryTracker


# --- is_boundary ---

def test_punctuation_boundaries():
    assert is_boundary(".")
    assert is_boundary(";")
    assert is_boundary("?")
    assert is_boundary("!")
    # Token with preceding text
    assert is_boundary("value.")
    assert is_boundary(" done!")


def test_double_newline():
    assert is_boundary("\n\n")
    assert is_boundary("text\n\nmore")


def test_logic_connectives():
    assert is_boundary("therefore")
    assert is_boundary(" Therefore")  # case-insensitive, whitespace
    assert is_boundary("however")
    assert is_boundary(" However,")  # with trailing comma
    assert is_boundary("so")
    assert is_boundary("but")
    assert is_boundary("thus")
    assert is_boundary("hence")
    assert is_boundary("because")


def test_structural_markers():
    assert is_boundary("Step")
    assert is_boundary("Step 1")
    assert is_boundary("Answer:")
    assert is_boundary("Therefore,")


def test_non_boundaries():
    assert not is_boundary("the")
    assert not is_boundary("123")
    assert not is_boundary(" x")
    assert not is_boundary("")
    assert not is_boundary("  ")
    assert not is_boundary("some")
    # "there" is not "therefore"
    assert not is_boundary("there")


# --- BoundaryTracker ---

def test_tracker_fires_at_k_min_with_boundary():
    tracker = BoundaryTracker(k_min=3, k_max=10)
    # 2 non-boundary tokens
    assert not tracker.step("the")
    assert not tracker.step("value")
    # 3rd token is boundary -> fires (chunk_len=3 >= k_min=3)
    assert tracker.step(".")


def test_tracker_does_not_fire_below_k_min():
    tracker = BoundaryTracker(k_min=5, k_max=100)
    # Boundary at token 2 should NOT fire
    assert not tracker.step("the")
    assert not tracker.step(".")  # boundary but chunk_len=2 < k_min=5


def test_tracker_forced_at_k_max():
    tracker = BoundaryTracker(k_min=5, k_max=8)
    for i in range(7):
        assert not tracker.step("word")
    # 8th token forces boundary regardless
    assert tracker.step("word")


def test_tracker_resets_after_fire():
    tracker = BoundaryTracker(k_min=2, k_max=100)
    assert not tracker.step("a")
    assert tracker.step(".")  # fires at chunk_len=2

    # After reset, need k_min tokens again
    assert not tracker.step("b")
    assert tracker.step("!")  # fires again at chunk_len=2


def test_tracker_manual_reset():
    tracker = BoundaryTracker(k_min=3, k_max=100)
    tracker.step("a")
    tracker.step("b")
    assert tracker.chunk_len == 2

    tracker.reset()
    assert tracker.chunk_len == 0
