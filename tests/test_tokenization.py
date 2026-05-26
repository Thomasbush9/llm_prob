import pytest

from llm_prob.tokenization import (
    BOS,
    CONTENT,
    EOS,
    ITOS,
    P,
    PAD,
    SEP,
    SPECIALS,
    STOI,
    VOCAB,
    build_sequence,
    encode_number,
    fmt,
)


# ---------- vocabulary ----------

def test_vocab_is_specials_then_content():
    assert VOCAB == SPECIALS + CONTENT


def test_vocab_has_no_duplicates():
    assert len(VOCAB) == len(set(VOCAB))


def test_stoi_itos_roundtrip():
    for tok in VOCAB:
        assert ITOS[STOI[tok]] == tok
    for i in range(len(VOCAB)):
        assert STOI[ITOS[i]] == i


def test_special_token_ids_are_distinct_and_in_range():
    ids = {PAD, BOS, EOS, SEP, P}
    assert len(ids) == 5
    for i in ids:
        assert 0 <= i < len(VOCAB)


def test_content_covers_digits_decimal_sign_comma():
    for ch in '0123456789.-,':
        assert ch in STOI


# ---------- fmt ----------

@pytest.mark.parametrize('x,decimals,expected', [
    (0.0, 2, '0.00'),
    (0.5, 2, '0.50'),
    (-1.234, 2, '-1.23'),
    (1.999, 2, '2.00'),
    (1.5, 3, '1.500'),
    (-0.0, 2, '-0.00'),
])
def test_fmt(x, decimals, expected):
    assert fmt(x, decimals) == expected


def test_fmt_default_is_two_decimals():
    assert fmt(3.14159) == '3.14'


# ---------- encode_number ----------

def test_encode_number_simple_decimal():
    assert encode_number('1.50') == [STOI['1'], STOI['.'], STOI['5'], STOI['0']]


def test_encode_number_negative():
    assert encode_number('-0.42') == [STOI['-'], STOI['0'], STOI['.'], STOI['4'], STOI['2']]


def test_encode_number_raises_on_unknown_char():
    with pytest.raises(KeyError):
        encode_number('abc')


def test_encode_number_decode_roundtrip():
    s = '-1.23'
    ids = encode_number(s)
    assert ''.join(ITOS[i] for i in ids) == s


# ---------- build_sequence ----------

def test_build_sequence_starts_with_bos_and_dist_token():
    toks, _ = build_sequence('normal', [0.0, 1.0], [0.5])
    assert toks[0] == BOS
    assert toks[1] == STOI['<NORMAL>']


def test_build_sequence_ends_with_eos():
    toks, _ = build_sequence('normal', [0.0, 1.0], [0.5])
    assert toks[-1] == EOS


def test_build_sequence_sample_start_lands_right_after_sep():
    toks, start = build_sequence('normal', [0.0, 1.0], [0.5, -0.5])
    assert toks[start - 1] == SEP
    # No SEP appears in the sample region.
    assert SEP not in toks[start:]


def test_build_sequence_has_one_p_per_param():
    toks, _ = build_sequence('normal', [0.0, 1.0], [0.5])
    assert toks.count(P) == 2

    toks3, _ = build_sequence('normal', [0.0, 1.0, 0.5], [0.5])
    assert toks3.count(P) == 3


def test_build_sequence_separates_samples_with_commas():
    samples = [0.1, 0.2, 0.3, 0.4]
    toks, start = build_sequence('normal', [0.0, 1.0], samples)
    # Strip the trailing EOS.
    sample_toks = toks[start:-1]
    comma_id = STOI[',']
    assert sample_toks.count(comma_id) == len(samples) - 1


def test_build_sequence_single_sample_has_no_comma():
    toks, start = build_sequence('normal', [0.0, 1.0], [0.5])
    sample_toks = toks[start:-1]
    assert STOI[','] not in sample_toks


def test_build_sequence_uniform_dist_uses_uniform_token():
    toks, _ = build_sequence('uniform', [0.0, 1.0], [0.5])
    assert toks[1] == STOI['<UNIFORM>']


def test_build_sequence_decimals_param_affects_width():
    toks2, _ = build_sequence('normal', [0.0, 1.0], [0.5], decimals=2)
    toks3, _ = build_sequence('normal', [0.0, 1.0], [0.5], decimals=3)
    # Each extra decimal adds one digit to each of the 2 params + 1 sample = 3 chars.
    assert len(toks3) - len(toks2) == 3


def test_build_sequence_decode_matches_expected_string():
    toks, start = build_sequence('normal', [0.0, 1.0], [0.5, -0.5])
    decoded = ''.join(ITOS[t] for t in toks)
    assert decoded == '<BOS><NORMAL><P>0.00<P>1.00<SEP>0.50,-0.50<EOS>'
    # And the start offset points at the '0' in "0.50".
    assert ITOS[toks[start]] == '0'


def test_build_sequence_params_encoded_in_order():
    toks, _ = build_sequence('normal', [1.23, -4.56], [0.0])
    decoded = ''.join(ITOS[t] for t in toks)
    assert '<P>1.23<P>-4.56<SEP>' in decoded
