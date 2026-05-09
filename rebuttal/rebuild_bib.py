#!/usr/bin/env python3
"""
rebuild_bib.py — 从 BibTeX_汇总.md 提取文献，按"一作+年份+短标题"重命名 key，
写入 reference.bib，并更新 HAG-DTA.tex 中的引用。
"""

import re, os

BASE = os.path.expanduser('~/HAG-DTA/1.HAG-DTA（BMC）')
BIB_MD = os.path.expanduser('~/HAG-DTA/rebuttal/BibTeX_汇总.md')
REF_BIB = os.path.join(BASE, 'reference.bib')
TEX_FILE = os.path.join(BASE, 'HAG-DTA.tex')

# ── Hardcoded per-entry author override (for accented names) ─────────
AUTHOR_OVERRIDE = {
    'DeepDTA':  'ozturk',
    'WideDTA':  'ozturk',
    'GAT':      'velickovic',
}
SHORT_TITLE_OVERRIDE = {
    'DPSP': 'dpsp',
}

def first_author_surname(authors_raw, old_key):
    if old_key in AUTHOR_OVERRIDE:
        return AUTHOR_OVERRIDE[old_key]
    first = authors_raw.split(',')[0]
    first = re.sub(r'\\[a-zA-Z"\']+', '', first)
    first = first.replace('{', '').replace('}', '')
    words = re.findall(r'[A-Za-z]+', first)
    return words[-1].lower() if words else 'unknown'


def short_title(title_raw, old_key):
    if old_key in SHORT_TITLE_OVERRIDE:
        return SHORT_TITLE_OVERRIDE[old_key]
    title = re.sub(r'\{|\}|\\[a-zA-Z]+', '', title_raw)
    words = re.findall(r'[a-zA-Z]{3,}', title)
    stop = {'the','for','with','based','using','from','of','in','on','to','a','an',
            'and','by','via','drug','target','prediction','predicting','method','model',
            'learning','network','deep','neural','protein','binding','affinity'}
    picked = [w.lower() for w in words if w.lower() not in stop][:2]
    return ''.join(picked) if picked else 'xx'


def parse_entries():
    with open(BIB_MD) as f:
        content = f.read()
    blocks = re.findall(r'```bibtex\n(.*?)```', content, re.DOTALL)
    entries = {}
    for block in blocks:
        block = block.strip()
        m = re.match(r'@(\w+)\{(\w+),', block)
        if not m:
            continue
        old = m.group(2)
        am = re.search(r'author\s*=\s*\{(.*?)\}', block, re.DOTALL)
        au = first_author_surname(am.group(1), old) if am else 'xx'
        ym = re.search(r'year\s*=\s*\{(\d{4})\}', block)
        yr = ym.group(1) if ym else '0000'
        tm = re.search(r'title\s*=\s*\{(.*?)\}', block, re.DOTALL)
        st = short_title(tm.group(1), old) if tm else 'xx'
        nk = f"{au}{yr}{st}"
        entries[old] = {'new_key': nk, 'block': block}
        print(f"  {old:25s} -> {nk}")
    return entries


def build_key_map(entries):
    km = {}
    for old, info in entries.items():
        km[old] = info['new_key']
    # Short-name overrides (keys used in tex without corresponding bib entry)
    short_map = {
        'GCNs':  km['GCN'],  'GINs':  km['GIN'],  'GATs':  km['GAT'],
        'GraphSAGE':        km['GraphSAGE'],
        'DiffPool':          km['DiffPool'],
        'KronRLS':           km['KronRLS'],
        'SimBoost':          km['SimBoost'],
        'DeepDTA':           km['DeepDTA'],
        'WideDTA':           km['WideDTA'],
        'AttentionDTA':      km['AttentionDTA'],
        'TransformerCPI':    km['TransformerCPI'],
        'MolTrans':          km['MolTrans'],
        'GraphDTA':          km['GraphDTA'],
        'MGraphDTA':         km['MGraphDTA'],
        'DGraphDTA':         km['DGraphDTA'],
        'MONN':              km['MONN'],
        'DPSP':              km['DPSP'],
        'NNPS':              km['NNPS'],
        'AttentionMGTDTA':   km['AttentionMGTDTA'],
        'GPCNDTA':           km['GPCNDTA'],
        'GEFormerDTA':       km['GEFormerDTA'],
        'GNPDTA':            km['GNPDTA'],
        'GLCNDTA':           km['GLCNDTA'],
        'CSCoDTA':           km['CSCoDTA'],
        'MDCTDTA':           km['MDCTDTA'],
        'WPGraphDTA':        km['WPGraphDTA'],
        'MEGDTA':            km['MEGDTA'],
        'VGAELDA':           km['VGAELDA'],
        'DAttProt':          km['DAttProt'],
        'NIMGSA':            km['NIMGSA'],
        'TLCrys':            km['TLCrys'],
    }
    km.update(short_map)
    return km


def write_bib(entries):
    with open(REF_BIB) as f:
        orig = f.read()
    orig_blocks = {}
    for m in re.finditer(r'(@\w+\{[^}]*\},.*?)(?=\n@\w+\{|$)', orig, re.DOTALL):
        block = m.group(0).strip()
        km = re.match(r'@\w+\{(\w+),', block)
        if km:
            orig_blocks[km.group(1)] = block
    汇总_old = set(entries.keys())
    汇总_new = set(e['new_key'] for e in entries.values())
    with open(REF_BIB, 'w') as f:
        for old, info in entries.items():
            block = re.sub(rf'@(\w+)\{{{old},', rf'@\1{{{info["new_key"]},', info['block'])
            f.write(block + '\n\n')
        for k, b in orig_blocks.items():
            if k not in 汇总_old and k not in 汇总_new:
                f.write(b + '\n\n')
    n_new = len(entries)
    n_kept = sum(1 for k in orig_blocks if k not in 汇总_old and k not in 汇总_new)
    print(f"  wrote {n_new} updated + {n_kept} originals")


def update_tex(key_map):
    with open(TEX_FILE) as f:
        tex = f.read()
    n = 0
    for old, new in sorted(key_map.items(), key=lambda x: -len(x[0])):
        pat = rf'(\\cite\w*\{{[^}}]*)' + re.escape(old) + r'([,}}])'
        tex, c = re.subn(pat, rf'\1{new}\2', tex)
        n += c
    with open(TEX_FILE, 'w') as f:
        f.write(tex)
    print(f"  {n} citation replacements")


def verify():
    with open(REF_BIB) as f:
        bib_keys = set(re.findall(r'@\w+\{(\w+)', f.read()))
    with open(TEX_FILE) as f:
        tex = f.read()
    tex_refs = set()
    for m in re.finditer(r'\\cite\w*\{([^}]+)\}', tex):
        for r in m.group(1).split(','):
            tex_refs.add(r.strip())
    missing = tex_refs - bib_keys
    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for k in sorted(missing):
            print(f"    {k}")
    else:
        print("\n  All citations matched — OK.")


def main():
    print("Parsing ...")
    entries = parse_entries()
    km = build_key_map(entries)
    write_bib(entries)
    update_tex(km)
    verify()

if __name__ == '__main__':
    main()
