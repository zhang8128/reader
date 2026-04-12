"""Tests for voice/gestures.py — classification logic only, no camera required."""

import pytest

from voice.gestures import classify_hand, Gesture


class FakeLandmark:
    def __init__(self, x, y, z=0):
        self.x, self.y, self.z = x, y, z


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_thumbs_up():
    """Thumb up, other fingers curled."""
    lm = [FakeLandmark(0, 0)] * 21
    # Thumb: tip (4) above MCP (2) — smaller y = higher
    lm[2] = FakeLandmark(0.4, 0.7)   # thumb MCP
    lm[3] = FakeLandmark(0.35, 0.5)  # thumb IP
    lm[4] = FakeLandmark(0.3, 0.3)   # thumb tip (y < MCP y = up)
    # Fingers curled: tip y >= PIP y
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        lm[mcp] = FakeLandmark(0.5, 0.7)
        lm[pip] = FakeLandmark(0.5, 0.6)
        lm[tip] = FakeLandmark(0.5, 0.75)  # tip below PIP = curled
    return lm


def _make_thumbs_down():
    """Thumb down, other fingers curled."""
    lm = [FakeLandmark(0, 0)] * 21
    # Thumb: tip (4) below MCP (2) — larger y = lower
    lm[2] = FakeLandmark(0.4, 0.3)   # thumb MCP
    lm[3] = FakeLandmark(0.35, 0.5)  # thumb IP
    lm[4] = FakeLandmark(0.3, 0.7)   # thumb tip (y > MCP y = down)
    # Fingers curled: tip y >= PIP y (hand inverted but still same rule)
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        lm[mcp] = FakeLandmark(0.5, 0.4)
        lm[pip] = FakeLandmark(0.5, 0.5)
        lm[tip] = FakeLandmark(0.5, 0.6)  # tip y > pip y = curled ✓
    return lm


def _make_open_palm():
    """All fingers extended — tips well above PIPs."""
    lm = [FakeLandmark(0, 0)] * 21
    lm[0] = FakeLandmark(0.5, 0.9)   # wrist
    # Thumb extended
    lm[2] = FakeLandmark(0.3, 0.7)
    lm[3] = FakeLandmark(0.2, 0.5)
    lm[4] = FakeLandmark(0.15, 0.3)
    # All fingers extended: tip y < pip y
    for mcp, pip, tip, xoff in [(5, 6, 8, 0.4), (9, 10, 12, 0.5), (13, 14, 16, 0.6), (17, 18, 20, 0.7)]:
        lm[mcp] = FakeLandmark(xoff, 0.7)
        lm[pip] = FakeLandmark(xoff, 0.5)
        lm[tip] = FakeLandmark(xoff, 0.2)  # tip y < pip y = extended ✓
    return lm


# ---------------------------------------------------------------------------
# Basic tests (from spec)
# ---------------------------------------------------------------------------

def test_thumbs_up():
    assert classify_hand(_make_thumbs_up()) == Gesture.THUMBS_UP


def test_thumbs_down():
    assert classify_hand(_make_thumbs_down()) == Gesture.THUMBS_DOWN


def test_open_palm():
    assert classify_hand(_make_open_palm()) == Gesture.OPEN_PALM


def test_no_landmarks_returns_none():
    assert classify_hand(None) is None


def test_all_same_position_returns_none():
    """All landmarks at the same point: fingers are 'curled' (tip y == pip y),
    but thumb tip y == thumb MCP y — neither up nor down → None."""
    lm = [FakeLandmark(0.5, 0.5)] * 21
    assert classify_hand(lm) is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_exactly_3_of_4_fingers_curled_is_thumbs_up():
    """3 of 4 fingers curled satisfies the threshold."""
    lm = _make_thumbs_up()
    # Uncurl the pinky: tip above its PIP
    lm[18] = FakeLandmark(0.7, 0.6)  # PIP
    lm[20] = FakeLandmark(0.7, 0.4)  # tip (y < PIP y = extended)
    assert classify_hand(lm) == Gesture.THUMBS_UP


def test_only_2_of_4_fingers_curled_returns_none():
    """Only 2 curled, 2 extended — neither threshold met → None."""
    lm = _make_thumbs_up()
    # Uncurl ring finger as well
    lm[14] = FakeLandmark(0.6, 0.6)  # PIP
    lm[16] = FakeLandmark(0.6, 0.4)  # tip extended
    lm[18] = FakeLandmark(0.7, 0.6)  # PIP
    lm[20] = FakeLandmark(0.7, 0.4)  # tip extended
    assert classify_hand(lm) is None


def test_exactly_3_of_4_fingers_extended_is_open_palm():
    """3 of 4 fingers extended satisfies the open-palm threshold."""
    lm = _make_open_palm()
    # Curl the pinky: tip below its PIP
    lm[18] = FakeLandmark(0.7, 0.5)  # PIP
    lm[20] = FakeLandmark(0.7, 0.7)  # tip (y > PIP y = curled)
    assert classify_hand(lm) == Gesture.OPEN_PALM


def test_only_2_of_4_fingers_extended_returns_none():
    """2 extended, 2 curled — not enough for either gesture."""
    lm = _make_open_palm()
    # Curl ring and pinky
    lm[14] = FakeLandmark(0.6, 0.5)  # ring PIP
    lm[16] = FakeLandmark(0.6, 0.7)  # ring tip curled
    lm[18] = FakeLandmark(0.7, 0.5)  # pinky PIP
    lm[20] = FakeLandmark(0.7, 0.7)  # pinky tip curled
    assert classify_hand(lm) is None


def test_thumbs_up_with_exactly_tip_equal_pip_counts_as_curled():
    """Tip y == PIP y counts as curled (>= boundary)."""
    lm = _make_thumbs_up()
    # Set one finger's tip exactly equal to PIP
    lm[6] = FakeLandmark(0.4, 0.6)  # index PIP
    lm[8] = FakeLandmark(0.4, 0.6)  # index tip — same y, counts as curled
    assert classify_hand(lm) == Gesture.THUMBS_UP


def test_thumb_level_with_mcp_is_ambiguous():
    """Thumb tip at same y as MCP — not up, not down → None."""
    lm = _make_thumbs_up()
    # Move thumb tip to same y as MCP
    lm[4] = FakeLandmark(0.3, lm[2].y)
    assert classify_hand(lm) is None


def test_thumbs_down_requires_curled_fingers():
    """Thumbs-down classification needs curled non-thumb fingers."""
    lm = _make_thumbs_down()
    # Extend all four fingers — should not be thumbs down
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        lm[pip] = FakeLandmark(0.5, 0.6)
        lm[tip] = FakeLandmark(0.5, 0.2)  # extended
    # Now fingers are extended — becomes open palm (thumb direction irrelevant)
    result = classify_hand(lm)
    assert result == Gesture.OPEN_PALM


def test_all_fingers_curled_thumb_level_returns_none():
    """Fingers curled but thumb at neutral position → None."""
    lm = [FakeLandmark(0.5, 0.5)] * 21
    # Thumb MCP and tip at same level
    lm[2] = FakeLandmark(0.4, 0.5)  # MCP
    lm[4] = FakeLandmark(0.4, 0.5)  # tip — equal
    # All other fingers curled
    for mcp, pip, tip in [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]:
        lm[pip] = FakeLandmark(0.5, 0.4)
        lm[tip] = FakeLandmark(0.5, 0.7)  # curled
    assert classify_hand(lm) is None


def test_gesture_enum_values():
    """Gesture enum has the expected string values."""
    assert Gesture.THUMBS_UP.value == "thumbs_up"
    assert Gesture.THUMBS_DOWN.value == "thumbs_down"
    assert Gesture.OPEN_PALM.value == "open_palm"
