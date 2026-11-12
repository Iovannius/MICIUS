"""Independent, bounded parser for M3 evidence; does not call the inference engine."""
import hashlib
import math
import struct


def need(ok, message):
    if not ok:
        raise ValueError(message)


class Reader:
    def __init__(self, raw):
        self.raw, self.pos = raw, 0

    def take(self, n):
        need(0 <= n <= len(self.raw) - self.pos, "Truncated snapshot")
        data = self.raw[self.pos:self.pos + n]
        self.pos += n
        return data

    def u32(self):
        return struct.unpack('<I', self.take(4))[0]

    def u64(self):
        return struct.unpack('<Q', self.take(8))[0]

    def text(self, limit):
        n = self.u32()
        need(n <= limit, "Snapshot string too long")
        return self.take(n).decode('utf-8', errors='strict')

    def end(self):
        need(self.pos == len(self.raw), "Trailing snapshot data")


def core(raw):
    r = Reader(raw)
    need(r.take(8) == b'MCSKV001', 'Unknown core state version')
    contract = r.text(64)
    need(len(contract) == 64 and all(c in '0123456789abcdef' for c in contract), 'Bad build identity')
    dims = [r.u32() for _ in range(10)]
    hidden, vocab, capacity, layers, heads, kvheads, head_dim, ff, mode, rope = dims
    need(all(x > 0 for x in dims[:8]) and hidden == heads * head_dim and heads % kvheads == 0,
         'Invalid model dimensions')
    eps, base = struct.unpack('<ff', r.take(8))
    need(math.isfinite(eps) and eps > 0 and math.isfinite(base) and base > 0, 'Bad model constants')
    split, bias, fp, seq, valid, generated = [r.u32() for _ in range(6)]
    need(split in (0, 1) and bias in (0, 1) and fp & ~0xffc0 == 0, 'Bad state flags')
    need(seq <= capacity and valid == int(seq != 0), 'Bad sequence length')
    active = seq * layers * kvheads * head_dim
    need(len(raw) - r.pos == (2 * active + (vocab if valid else 0)) * 4, 'Active KV/logits byte count differs')
    need(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw[r.pos:])), 'Nonfinite KV/logits')
    return {'contract': contract, 'dimensions': dims, 'sequence': seq, 'generated_counter': generated,
            'active_kv_bytes': active * 8, 'logits_bytes': vocab * 4 if valid else 0,
            'numerical_sha256': hashlib.sha256(raw[:144] + b'\0' * 4 + raw[148:]).hexdigest()}


def snapshot(raw):
    need(len(raw) >= 64 and hashlib.sha256(raw[:-64]).hexdigest().encode() == raw[-64:], 'Snapshot checksum mismatch')
    r = Reader(raw[:-64])
    need(r.take(8) == b'MCSCHAT1', 'Unknown chat snapshot version')
    model, arch, conf = r.text(64), r.text(128), r.text(16384)
    need(len(model) == 64 and all(c in '0123456789abcdef' for c in model), 'Bad model identity')
    config = {}
    for line in conf.splitlines():
        key, sep, value = line.partition('=')
        need(sep and key not in config, 'Bad stored configuration')
        config[key] = value
    need(set(config) == {'version', 'temperature', 'top_k', 'top_p', 'repeat_penalty', 'repeat_last_n', 'seed', 'max_new_tokens'}
         and config['version'] == '1', 'Incomplete stored configuration')
    random, turns, count = r.u64(), r.u64(), r.u32()
    need(count <= (len(r.raw) - r.pos) // 4, 'Token history too long')
    tokens = [r.u32() for _ in range(count)]
    state = r.take(r.u64())
    r.end()
    decoded = core(state)
    need(decoded['sequence'] == count and all(t < decoded['dimensions'][1] for t in tokens), 'Token/KV mismatch')
    need((turns == 0) == (count == 0) and turns <= count, 'Bad saved turn count')
    return {'model_sha256': model, 'architecture': arch, 'config': config, 'rng': random, 'turns': turns,
            'tokens': tokens, 'core': decoded}, state


def same_reset(a, b):
    """Only the V1 lifetime diagnostic counter is allowed to survive reset."""
    aa, _ = snapshot(a)
    bb, _ = snapshot(b)
    aa['core'].pop('generated_counter')
    bb['core'].pop('generated_counter')
    need(aa == bb, 'Reset did not reproduce history/RNG/config/KV/logits')
