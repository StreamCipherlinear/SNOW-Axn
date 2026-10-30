"""Print every passage around the word "correl" in a text file.

Inputs: a file name on the command line.
Outputs: byte offsets and excerpts around each match.
Paper object: none; development aid for reading drafts and reviews.
"""

import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
s=open(sys.argv[1],encoding='utf-8').read()
# keep only plausible text: printable ascii + latin1 letters
def clean(t):
    return '\u0000'.join(ch for ch in t if 32 <= ord(ch) < 127 or ch in '—-')
for m in re.finditer(r'correl', s):
    seg = s[max(0,m.start()-250):m.start()+450]
    seg = re.sub(r'[^\x20-\x7e\n]', '', seg)
    print('---', m.start(), '---')
    print(seg)
