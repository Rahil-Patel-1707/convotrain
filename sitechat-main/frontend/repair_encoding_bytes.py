from pathlib import Path

path = Path("landing.html")
data = path.read_bytes()

# Replace the UTF-8 byte sequences representing the mojibake
# with the correct UTF-8 bytes.
replacements = {
    b"\xc3\x82\xc2\xb7": "·".encode("utf-8"),
    b"\xc3\xa2\xe2\x82\xac\xe2\x80\x94": "—".encode("utf-8"),
    b"\xc3\xa2\xe2\x82\xac\xe2\x80\x93": "–".encode("utf-8"),
    b"\xc3\xa2\xe2\x80\xa0\xe2\x86\x92": "→".encode("utf-8"),
    b"\xc3\xa2\xe2\x82\xac\xc2\xa6": "…".encode("utf-8"),
    b"\xc3\x82\xc2\xa9": "©".encode("utf-8"),
}

total = 0

for old, new in replacements.items():
    count = data.count(old)
    if count:
        print(f"{old!r}: {count}")
        data = data.replace(old, new)
        total += count

path.write_bytes(data)

print(f"Total byte replacements: {total}")
