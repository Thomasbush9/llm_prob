"""Character-level vocabulary and sequence builder for distribution-modelling prompts.

A sequence looks like:
    <BOS><NORMAL><P>0.00<P>1.00<SEP>0.30,-1.04,...,−0.85<EOS>

The integer returned alongside the token list is the index of the first token
inside the sample region (i.e. immediately after <SEP>) — used to build the
loss mask in `data.collate`.
"""

SPECIALS = ['<PAD>', '<BOS>', '<EOS>', '<SEP>', '<P>', '<NORMAL>', '<UNIFORM>']
CONTENT = list('0123456789.-,')
VOCAB = SPECIALS + CONTENT
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}

PAD = STOI['<PAD>']
BOS = STOI['<BOS>']
EOS = STOI['<EOS>']
SEP = STOI['<SEP>']
P = STOI['<P>']


def fmt(x, decimals=2):
    return f"{x:.{decimals}f}"


def encode_number(s):
    return [STOI[c] for c in s]


def build_sequence(dist, params, samples, decimals=2):
    toks = [BOS, STOI[f'<{dist.upper()}>']]
    for p in params:
        toks.append(P)
        toks.extend(encode_number(fmt(p, decimals)))
    toks.append(SEP)
    sample_start = len(toks)
    for i, x in enumerate(samples):
        toks.extend(encode_number(fmt(x, decimals)))
        if i < len(samples) - 1:
            toks.append(STOI[','])
    toks.append(EOS)
    return toks, sample_start
