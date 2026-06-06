#!/usr/bin/env python3
"""Read-only vault audit. Run from anywhere:
    python3 docs/superpowers/tools/vault_audit.py
Reports dangling wikilinks, orphan notes (no inbound links), and frontmatter gaps.
Exit code 0 iff there are no dangling links, else 1."""
from __future__ import annotations
import re, sys
from pathlib import Path

VAULT = Path(__file__).resolve().parents[3]      # docs/superpowers/tools -> vault root
SKIP_DIRS = {'.git', '.obsidian', 'docs', '.vscode', 'Assets', 'Templates'}
SKIP_FILES = {'CLAUDE.md', 'VAULT-INSTRUCTIONS.md', 'README.md'}

# Target = chars up to a heading (#), an alias pipe (| or escaped \|), or closing ]].
# Handles Obsidian table-cell alias syntax [[Target\|display]] where | is escaped as \|.
wikilink_re = re.compile(r'\[\[([^\[\]|#\\]+)(?:\\?[#|][^\[\]]*)?\]\]')
fm_re = re.compile(r'\A---\s*\n(.*?)\n---\s*\n', re.DOTALL)

def iter_md():
    for p in sorted(VAULT.rglob('*.md')):
        rel = p.relative_to(VAULT)
        if set(rel.parts) & SKIP_DIRS or p.name in SKIP_FILES:
            continue
        yield p

def get_fm(text):
    m = fm_re.match(text)
    return m.group(1) if m else None

def fm_field(fm, key):
    if fm is None:
        return None
    m = re.search(rf'^{key}\s*:\s*(.*)$', fm, re.MULTILINE)
    return m.group(1).strip() if m else None

def aliases_of(fm):
    raw = fm_field(fm, 'aliases')
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith('['):
        raw = raw[1:-1]
    return [a.strip().strip('"\'') for a in raw.split(',') if a.strip()]

def main():
    files = list(iter_md())
    resolve = {}                 # lowercase name/alias -> Path
    notes = {}                   # Path -> info dict
    for p in files:
        text = p.read_text(encoding='utf-8', errors='replace')
        notes[p] = {'fm': get_fm(text), 'text': text, 'inbound': set()}
    # Resolution priority mirrors Obsidian: an exact filename (stem) match wins over any
    # alias, so register all stems first, then aliases only for still-unclaimed names.
    for p in files:
        resolve.setdefault(p.stem.lower(), p)
    for p in files:
        for a in aliases_of(notes[p]['fm']):
            resolve.setdefault(a.lower(), p)

    dangling = []
    for p, info in notes.items():
        for m in wikilink_re.finditer(info['text']):
            if m.start() > 0 and info['text'][m.start() - 1] == '!':
                continue                       # image/embed, not a link
            target = m.group(1).strip()
            if not target:
                continue
            key = target.lower()
            base = key.rstrip('/').split('/')[-1]
            tgt = resolve.get(key) or resolve.get(base)
            if tgt is None:
                dangling.append((str(p.relative_to(VAULT)), target))
            elif tgt != p:
                notes[tgt]['inbound'].add(p)

    orphans = [str(p.relative_to(VAULT))
               for p, info in notes.items()
               if not info['inbound'] and fm_field(info['fm'], 'type') != 'moc']

    fm_gaps = []
    for p, info in notes.items():
        rel = str(p.relative_to(VAULT))
        if info['fm'] is None:
            fm_gaps.append((rel, 'no frontmatter')); continue
        for key in ('type', 'status'):
            if fm_field(info['fm'], key) is None:
                fm_gaps.append((rel, f'missing {key}'))

    print(f"# Vault audit — {len(files)} notes\n")
    print(f"## Dangling links ({len(dangling)})")
    for f, t in dangling:
        print(f"  - {f}  ->  [[{t}]]")
    print(f"\n## Orphans / no inbound links ({len(orphans)})")
    for f in orphans:
        print(f"  - {f}")
    print(f"\n## Frontmatter gaps ({len(fm_gaps)})")
    for f, why in fm_gaps:
        print(f"  - {f}: {why}")
    ok = not dangling
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(dangling={len(dangling)}, orphans={len(orphans)}, fm_gaps={len(fm_gaps)})")
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
