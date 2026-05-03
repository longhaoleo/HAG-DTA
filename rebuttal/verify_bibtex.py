#!/usr/bin/env python3
"""
verify_bibtex.py — 快速验证 BibTeX 汇总中的 DOI 是否正确

对每条文献：用 DOI 访问 Crossref API，比对返回标题与预期标题的相似度。
相似度低于阈值（默认 0.6）的标记为 MISMATCH。

Usage:
  python3 rebuttal/verify_bibtex.py
"""
import urllib.request, json, time, re
from difflib import SequenceMatcher

# ── 文献列表：key, expected_title_prefix, DOI ──
ENTRIES = [
    ("KronRLS",         "Toward more realistic drug",       "10.1093/bib/bbu010"),
    ("SimBoost",        "SimBoost: a read-across approach", "10.1186/s13321-017-0209-z"),
    ("DeepDTA",         "DeepDTA: deep drug",               "10.1093/bioinformatics/bty593"),
    ("WideDTA",         "WideDTA: prediction",              None),  # arXiv
    ("AttentionDTA",    "AttentionDTA: Drug",               "10.1109/TCBB.2022.3170365"),
    ("TransformerCPI",  "TransformerCPI: improving",        "10.1093/bioinformatics/btaa524"),
    ("MolTrans",        "MolTrans: Molecular Interaction",  "10.1093/bioinformatics/btaa880"),
    ("GraphDTA",        "GraphDTA: predicting drug",        "10.1093/bioinformatics/btaa921"),
    ("MGraphDTA",       "MGraphDTA: deep multiscale",       "10.1039/d1sc05180f"),
    ("DGraphDTA",       "Drug-target affinity prediction using graph neural network and contact maps", "10.1039/d0ra02297g"),
    ("AttentionMGTDTA", "AttentionMGT-DTA: A multi-modal",  "10.1016/j.neunet.2023.11.018"),
    ("GPCNDTA",         "GPCNDTA: Prediction of drug",      "10.1016/j.compbiomed.2023.107512"),
    ("GEFormerDTA",     "GEFormerDTA: drug target",         "10.1038/s41598-024-57879-1"),
    ("GNPDTA",          "Graph neural pre-training",        "10.3389/fgene.2024.1452339"),
    ("GLCNDTA",         "Drug-target affinity prediction with extended graph", "10.1186/s12859-024-05698-6"),
    ("CSCoDTA",         "Predicting drug-target binding affinity with cross-scale", "10.1093/bib/bbad516"),
    ("MDCTDTA",         "Drug-target binding affinity prediction model based on multi-scale diffusion", "10.1016/j.eswa.2024.124647"),
    ("WPGraphDTA",      "Drug-target binding affinity prediction based on power graph", "10.1186/s12920-024-02073-5"),
    ("MEGDTA",          "MEGDTA: multi-modal drug",         "10.1186/s12864-025-11943-w"),
    ("MONN",            "MONN: A Multi-objective",          "10.1016/j.cels.2020.03.002"),
    ("DPSP",            "DPSP: a multimodal deep learning", "10.1093/bioadv/vbad110"),
    ("NNPS",            "A neural network-based method for polypharmacy", "10.1186/s12859-021-04298-y"),
    ("VGAELDA",         "A representation learning model based on variational inference", "10.1186/s12859-021-04073-z"),
    ("DAttProt",        "An Interpretable Double-Scale Attention Model", "10.3389/fgene.2022.885627"),
    ("NIMGSA",          "Predicting miRNA-Disease Association", "10.3390/biom12010064"),
    ("TLCrys",          "TLCrys: Transfer Learning",        "10.3390/ijms23020972"),
]

THRESHOLD = 0.6
pass_count = 0
fail_count = 0
arxiv_count = 0

print(f"{'Key':<20} {'Status':<12} {'Title (first 80 chars)'}")
print("-" * 112)

for key, expected, doi in ENTRIES:
    if doi is None:
        print(f"{key:<20} {'arXiv':<12} (cannot verify via Crossref, check manually)")
        arxiv_count += 1
        continue

    url = f"https://api.crossref.org/works/{doi}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HAG-DTA/1.0"})
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read())
        title = data["message"]["title"][0]
    except Exception as e:
        print(f"{key:<20} {'ERROR':<12} {str(e)[:60]}")
        fail_count += 1
        continue

    # 忽略大小写和破折号差异，直接比对实际标题是否以期望的前缀开头
    actual_clean = title.lower().replace('\u2013', '-').replace('\u2014', '-')
    expected_clean = expected.lower().replace('\u2013', '-').replace('\u2014', '-')
    ratio = SequenceMatcher(None, expected_clean, actual_clean[:len(expected_clean)+20]).ratio()
    if ratio >= THRESHOLD:
        print(f"{key:<20} {'✅ PASS':<12} {title[:80]}")
        pass_count += 1
    else:
        print(f"{key:<20} {'❌ MISMATCH':<12} {title[:80]}")
        print(f"{'':>20} {'':>12} expected: {expected[:80]}")
        fail_count += 1
    time.sleep(0.5)

print("-" * 112)
print(f"Results: {pass_count} pass, {fail_count} fail, {arxiv_count} arXiv (manual)")
