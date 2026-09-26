from pathlib import Path

path = Path("landing.html")
text = path.read_text(encoding="utf-8-sig")

# Build malformed sequences by code point.
# These correspond to the mojibake visible in the original file.
bad_em_dash = "".join(chr(x) for x in [0x00E2, 0x20AC, 0x2014])
bad_en_dash = "".join(chr(x) for x in [0x00E2, 0x20AC, 0x2013])
bad_arrow   = "".join(chr(x) for x in [0x00E2, 0x2020, 0x2192])
bad_bullet  = "".join(chr(x) for x in [0x00C2, 0x00B7])
bad_copy    = "".join(chr(x) for x in [0x00C2, 0x00A9])

good_em_dash = chr(0x2014)
good_en_dash = chr(0x2013)
good_arrow   = chr(0x2192)
good_bullet  = chr(0x00B7)
good_copy    = chr(0x00A9)

replacements = [
    (bad_em_dash, good_em_dash),
    (bad_en_dash, good_en_dash),
    (bad_arrow, good_arrow),
    (bad_bullet, good_bullet),
    (bad_copy, good_copy),
]

total = 0

for old, new in replacements:
    count = text.count(old)
    if count:
        print(f"Replacing {count} occurrence(s)")
        text = text.replace(old, new)
        total += count

path.write_text(text, encoding="utf-8")

print(f"Total replacements: {total}")
