from pathlib import Path

path = Path("landing.html")
text = path.read_text(encoding="utf-8")

replacements = {
    # Exact code-point sequences found by the diagnostic.
    "\u00e2\u20ac\u201d": "\u2014",  # â€” -> —
    "\u00e2\u20ac\u201c": "\u2013",  # â€“ -> –
    "\u00e2\u2020\u2019": "\u2192",  # â†’ -> →
}

total = 0

for old, new in replacements.items():
    count = text.count(old)
    if count:
        print(f"Replacing {repr(old)} -> {repr(new)}: {count}")
        text = text.replace(old, new)
        total += count

path.write_text(text, encoding="utf-8")

print(f"Total replacements: {total}")
