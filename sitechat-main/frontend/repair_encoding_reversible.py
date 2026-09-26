from pathlib import Path

path = Path("landing.html")
text = path.read_text(encoding="utf-8-sig")

# Repeatedly reverse UTF-8 -> Windows-1252 mojibake,
# but only when the resulting text becomes valid UTF-8.
def repair_mojibake(value):
    for _ in range(3):
        try:
            repaired = value.encode("latin1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break

        if repaired == value:
            break

        value = repaired

    return value

lines = text.splitlines(keepends=True)

changed = 0

for i, line in enumerate(lines):
    if any(ord(c) in range(0x80, 0x100) for c in line):
        repaired = repair_mojibake(line)

        if repaired != line:
            lines[i] = repaired
            changed += 1
            print(f"Repaired line {i + 1}")

text = "".join(lines)

path.write_text(text, encoding="utf-8")

print(f"Changed lines: {changed}")
print("Repair complete.")
