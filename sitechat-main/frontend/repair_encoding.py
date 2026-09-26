from pathlib import Path

path = Path("landing.html")
text = path.read_text(encoding="utf-8-sig")

replacements = {
    "\u00c2\u00b7": "\u00b7",
    "\u00c2\u00a9": "\u00a9",
    "\u00e2\u2020\u0092": "\u2192",
    "\u00e2\u20ac\u2014": "\u2014",
    "\u00e2\u20ac\u2013": "\u2013",
    "\u00e2\u20ac\u00a6": "\u2026",
}

total = 0

for old, new in replacements.items():
    count = text.count(old)
    if count:
        print(f"Replacing {count} occurrence(s)")
        text = text.replace(old, new)
        total += count

path.write_text(text, encoding="utf-8")

print(f"Total replacements: {total}")
print("Done")
