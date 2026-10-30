"""Print the context around regular expressions in a text file.

Inputs: a file name and a |-separated pattern list on the command line.
Outputs: for each match, a short prefix and a longer suffix of the text.
Paper object: none; development aid for reading drafts and reviews.
"""

import re, sys
s=open(sys.argv[1],encoding='utf-8').read()
pats=sys.argv[2].split('|')
for pat in pats:
    print('#### ', pat)
    for m in re.finditer(pat, s, re.I):
        a=max(0,m.start()-300); b=min(len(s), m.end()+700)
        print('---@',m.start(),'---')
        print(s[a:b].replace('\n',' '))
        print()
