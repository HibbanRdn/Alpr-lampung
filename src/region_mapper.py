"""Lampung license plate region mapping utilities."""

from __future__ import annotations

import re
from typing import Any


REGION_MAP = {
    "A": "Kota Bandar Lampung",
    "B": "Kota Bandar Lampung",
    "C": "Kota Bandar Lampung",
    "Y": "Kota Bandar Lampung",
    "F": "Kota Metro",
    "D": "Kab. Lampung Selatan",
    "E": "Kab. Lampung Selatan",
    "O": "Kab. Lampung Selatan",
    "M": "Kab. Lampung Barat",
    "G": "Kab. Lampung Tengah",
    "H": "Kab. Lampung Tengah",
    "I": "Kab. Lampung Tengah",
    "R": "Kab. Pesawaran",
    "U": "Kab. Pringsewu",
    "X": "Kab. Pesisir Barat",
    "N": "Kab. Lampung Timur",
    "P": "Kab. Lampung Timur",
    "J": "Kab. Lampung Utara",
    "K": "Kab. Lampung Utara",
    "V": "Kab. Tanggamus",
    "Z": "Kab. Tanggamus",
    "W": "Kab. Way Kanan",
    "S": "Kab. Tulang Bawang",
    "T": "Kab. Tulang Bawang",
    "Q": "Kab. Tulang Bawang Barat",
    "L": "Kab. Mesuji",
}


def normalize_plate_text(text: Any) -> str:
    """Normalize plate text into compact uppercase alphanumeric form."""
    if text is None:
        return ""
    normalized = str(text).upper()
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    if normalized.startswith(("8E", "6E")):
        normalized = "BE" + normalized[2:]
    elif normalized.startswith("B3"):
        normalized = "BE" + normalized[2:]
    elif normalized.startswith("13E"):
        normalized = "BE" + normalized[3:]
    return normalized


def format_plate_text(text: Any) -> str:
    """Format normalized plate text into `BE 1234 ABC` when possible."""
    normalized = normalize_plate_text(text)
    match = re.match(r"^(BE)(\d{1,4})([A-Z]{1,3})$", normalized)
    if not match:
        return normalized
    return f"{match.group(1)} {match.group(2)} {match.group(3)}"


def is_valid_lampung_plate(text: Any) -> bool:
    """Return whether text follows the supported Lampung plate pattern."""
    return bool(re.match(r"^BE\d{1,4}[A-Z]{1,3}$", normalize_plate_text(text)))


def extract_region_code(plate_text: Any) -> str:
    """Extract the first suffix letter after the plate number."""
    normalized = normalize_plate_text(plate_text)
    match = re.match(r"^BE\d{1,4}([A-Z]{1,3})$", normalized)
    return match.group(1)[0] if match else ""


def identify_region_from_code(code: Any) -> str:
    """Map a Lampung region code to city/regency name."""
    normalized = str(code or "").upper().strip()
    return REGION_MAP.get(normalized[:1], "Unknown")


def map_plate_to_region(plate_text: Any) -> dict[str, str | bool]:
    """Return normalized plate, region code, region name, and validity."""
    normalized = normalize_plate_text(plate_text)
    code = extract_region_code(normalized)
    return {
        "plate_norm": normalized,
        "plate_formatted": format_plate_text(normalized),
        "region_code": code,
        "region_name": identify_region_from_code(code) if code else "Unknown",
        "is_valid": is_valid_lampung_plate(normalized),
    }

