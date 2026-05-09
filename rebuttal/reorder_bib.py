#!/usr/bin/env python3
import re, os

bib_path = os.path.expanduser('~/HAG-DTA/1.HAG-DTA（BMC）/reference.bib')
with open(bib_path) as f:
    content = f.read()

blocks = re.findall(r'(@\w+\{[^}]*\},.*?)(?=\n@\w+\{|$)', content, re.DOTALL)

masumshah_blocks = []
other_blocks = []
for block in blocks:
    km = re.match(r'@\w+\{(\w+),', block)
    if km and 'masumshah' in km.group(1).lower():
        masumshah_blocks.append(block.strip())
    else:
        other_blocks.append(block.strip())

# Reorder: other blocks first, then ### comment + masumshah blocks
result = '\n\n'.join(other_blocks)
result += '\n\n'
result += '### Reviewer-recommended citations (R1)\n\n'
result += '\n\n'.join(masumshah_blocks)
result += '\n'

with open(bib_path, 'w') as f:
    f.write(result)

print(f"Moved {len(masumshah_blocks)} Masumshah entries to end with ### marker")
