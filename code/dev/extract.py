"""Extract the text strings of a PDF file into a plain text file.

Inputs: an input PDF and an output text path on the command line.
Outputs: the concatenated, unescaped text strings written to the output path.
Paper object: none; development aid for comparing against the design document.
"""

import re, zlib, sys, os
p = sys.argv[1]
d = open(p,'rb').read()
out=[]
for m in re.finditer(rb'stream\r?\n', d):
    start=m.end(); e=d.find(b'endstream', start)
    if e<0: continue
    try: dec=zlib.decompress(d[start:e])
    except Exception: continue
    out.append(dec)
blob=b'\n'.join(out)
def unescape(b):
    res=bytearray(); i=0
    while i < len(b):
        c=b[i]
        if c==0x5c and i+1<len(b):
            n=b[i+1]
            if 0x30<=n<=0x37:
                j=i+1; oct_=b''
                while j<len(b) and 0x30<=b[j]<=0x37 and len(oct_)<3:
                    oct_+=bytes([b[j]]); j+=1
                res.append(int(oct_,8)); i=j; continue
            mapping={0x6e:10,0x72:13,0x74:9,0x62:8,0x66:12,0x28:40,0x29:41,0x5c:92}
            res.append(mapping.get(n,n)); i+=2; continue
        res.append(c); i+=1
    return bytes(res)
txt=[]
for m in re.finditer(rb'\((?:\\.|[^\\()])*\)', blob):
    txt.append(unescape(m.group()[1:-1]))
s=b''.join(txt)
s=s.decode('latin-1')
s=s.replace('\x1c','fi').replace('\x1d','fl').replace('\x0b','ffi')
s=re.sub(r'[ \t]+',' ',s)
open(sys.argv[2],'w',encoding='utf-8').write(s)
print('written', len(s), 'chars')
