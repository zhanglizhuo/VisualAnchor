#!/bin/bash
# Generate blind (double-anonymized) version from canonical VisualAnchor.tex
set -e
cd "$(dirname "$0")"

cp VisualAnchor.tex VisualAnchor_blind.tex

# 1. shortauthors
sed -i 's/\\shortauthors{Ma and Zhang}/\\shortauthors{}/' VisualAnchor_blind.tex

# 2. Replace author block using a Python script that avoids backslash-in-regex issues
python3 << 'PYEOF'
text = open('VisualAnchor_blind.tex').read()

BS = '\\'  # single backslash, not an escape

# Author block replacement
import re
old = re.escape(BS + 'author[1,2]{Yan Ma}')
# Match from \author[1,2]{Yan Ma} to \nonumnote{...} (inclusive)
idx = text.find(BS + 'author[1,2]{Yan Ma}')
if idx < 0:
    print("ERROR: author block not found")
    exit(1)

end_marker = BS + 'nonumnote{This research received no external funding.}'
idx2 = text.find(end_marker, idx)
if idx2 < 0:
    print("ERROR: nonumnote not found")
    exit(1)
idx2 += len(end_marker)

anon_block = (BS + 'author[1]{Anonymous Author}\n' +
              BS + 'author[2]{Anonymous Author}\n\n' +
              BS + 'affiliation[1]{organization={Anonymous Institution},\n' +
              '    city={Anonymous},\n' +
              '    country={Anonymous}}\n\n' +
              BS + 'affiliation[2]{organization={Anonymous Institution},\n' +
              '    city={Anonymous},\n' +
              '    country={Anonymous}}\n\n' +
              '% Author details withheld for double-blind review.')

text = text[:idx] + anon_block + text[idx2:]

# "companion paper" → "prior work"
text = text.replace(
    'The companion paper~' + BS + 'cite{ma2026agreement}',
    'Prior work~' + BS + 'cite{ma2026agreement}')
text = text.replace(
    'The companion paper reports',
    'Prior work reports')
text = text.replace(
    'introduced in our companion study',
    'introduced in prior work~' + BS + 'cite{ma2026agreement}')

# GitHub URL
text = text.replace(
    'The experiment code, prompt templates, per-image MLLM predictions, and all result files are available at ' + BS + 'url{https://github.com/zhanglizhuo/VisualAnchor}.',
    'Code and result files will be made available upon acceptance.')

# Remove biographies (from \bio{} to \endbio)
while BS + 'bio{}' in text:
    i = text.find(BS + 'bio{}')
    j = text.find(BS + 'endbio', i)
    text = text[:i] + text[j+len(BS+'endbio'):]

# Use blind bib
text = text.replace(BS + 'bibliography{refs}', BS + 'bibliography{refs_blind}')

open('VisualAnchor_blind.tex', 'w').write(text)
print('OK')
PYEOF

# 3. Compile
pdflatex -interaction=nonstopmode VisualAnchor_blind
bibtex VisualAnchor_blind
pdflatex -interaction=nonstopmode VisualAnchor_blind
pdflatex -interaction=nonstopmode VisualAnchor_blind

echo "✓ VisualAnchor_blind.pdf generated"
