#!/usr/bin/env python3
"""
fix_bib.py — 将 HAG-DTA.tex 中的短引用名映射到 reference.bib 已有的 key，
新增文献追加到 bib 末尾。
"""

import re, os

BASE = os.path.expanduser('~/HAG-DTA/1.HAG-DTA（BMC）')
BIB_MD = os.path.expanduser('~/HAG-DTA/rebuttal/BibTeX_汇总.md')
REF_BIB = os.path.join(BASE, 'reference.bib')
TEX = os.path.join(BASE, 'HAG-DTA.tex')

# ── 已有的 bib 条目 ──
with open(REF_BIB) as f:
    bib_text = f.read()
bib_keys = set(re.findall(r'@\w+\{(\w+),', bib_text))

# ── 短名 → bib key 映射（手工确认） ──
SHORT_TO_BIB = {
    'KronRLS':         'pahikkala2015toward',
    'SimBoost':        'he2017simboost',
    'DeepDTA':         'ozturk2018deepdta',
    'WideDTA':         'ozturk2019widedta',
    'TransformerCPI':  'chen2020transformercpi',
    'MolTrans':        'huang2021moltrans',
    'GraphDTA':        'nguyen2021graphdta',
    'MGraphDTA':       'yang2022mgraphdta',
    'GraphSAGE':       'hamilton2017inductive',
    'DiffPool':        'ying2018hierarchical',
    'GCN':             'kipf2016semi',
    'GIN':             'xu2018powerful',
    'GAT':             'velivckovic2017graph',
    'DGraphDTA':       'jiang2022sequence',
    # below: not in old bib — will be added
}

# ── 从 汇总 解析所有条目 ──
with open(BIB_MD) as f:
    md = f.read()
汇总_blocks = {}
for block in re.findall(r'```bibtex\n(.*?)```', md, re.DOTALL):
    block = block.strip()
    m = re.match(r'@(\w+)\{(\w+),', block)
    if m:
        汇总_blocks[m.group(2)] = block

# ── 新增不在 bib 中的文献 ──
new_entries = []
for short_key, block in 汇总_blocks.items():
    if short_key in SHORT_TO_BIB:
        continue  # already mapped to existing bib key
    # Generate new key: first author + year + short title
    am = re.search(r'author\s*=\s*\{(.*?)\}', block, re.DOTALL)
    ym = re.search(r'year\s*=\s*\{(\d{4})\}', block)
    tm = re.search(r'title\s*=\s*\{(.*?)\}', block, re.DOTALL)
    if not am or not ym:
        continue
    # First author surname
    first = am.group(1).split(',')[0]
    first = re.sub(r'\\[a-zA-Z"\']+', '', first).replace('{','').replace('}','')
    words = re.findall(r'[A-Za-z]+', first)
    au = words[-1].lower() if words else 'xx'
    # Short title
    title = re.sub(r'\{|\}|\\[a-zA-Z]+', '', tm.group(1))
    tw = re.findall(r'[a-zA-Z]{3,}', title)
    stop = {'the','for','with','based','using','from','of','in','on','to','a','an',
            'and','by','via','drug','target','prediction','predicting','method','model',
            'learning','network','deep','neural','protein','binding','affinity'}
    picked = [w.lower() for w in tw if w.lower() not in stop][:2]
    st = ''.join(picked) if picked else 'xx'
    nk = f"{au}{ym.group(1)}{st}"
    # Ensure unique
    if nk in bib_keys or nk in SHORT_TO_BIB.values():
        nk += 'a'
    SHORT_TO_BIB[short_key] = nk
    new_entries.append((nk, block))
    print(f"  NEW: {short_key:25s} -> {nk}")

# Also map plural forms
for k, v in list(SHORT_TO_BIB.items()):
    SHORT_TO_BIB[k + 's'] = v

# ── 写入 bib ──
with open(REF_BIB, 'a') as f:
    f.write('\n')
    for nk, block in new_entries:
        new_block = re.sub(rf'@(\w+)\{{{list(汇总_blocks.keys())[list(汇总_blocks.values()).index(block)]},', 
                           rf'@\1{{{nk},', block)
        f.write(new_block + '\n\n')

print(f"\n  Added {len(new_entries)} new entries to reference.bib")

# ── 更新 tex ──
with open(TEX) as f:
    tex = f.read()
n = 0
for old, new in sorted(SHORT_TO_BIB.items(), key=lambda x: -len(x[0])):
    pat = rf'(\\cite\w*\{{[^}}]*)' + re.escape(old) + r'([,}}])'
    tex, c = re.subn(pat, rf'\1{new}\2', tex)
    n += c
with open(TEX, 'w') as f:
    f.write(tex)
print(f"  {n} tex citation replacements")

# ── 验证 ──
with open(REF_BIB) as f:
    bib_keys_final = set(re.findall(r'@\w+\{(\w+),', f.read()))
with open(TEX) as f:
    tex_final = f.read()
tex_refs = set()
for m in re.finditer(r'\\cite\w*\{([^}]+)\}', tex_final):
    for r in m.group(1).split(','):
        tex_refs.add(r.strip())
missing = tex_refs - bib_keys_final
if missing:
    print(f"\n  MISSING ({len(missing)}):")
    for k in sorted(missing): print(f"    {k}")
else:
    print("  All citations OK.")
