import numpy as np

from parakeet_nemo_asr_rocm.timestamps.word_timestamps import get_word_timestamps


class _Tensor:
    def __init__(self, arr):
        self.arr = np.array(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr


class _Hypo:
    def __init__(self, ids, times, offset=0.0):
        self.y_sequence = _Tensor(ids)
        self.timestamp = _Tensor(times)
        self.start_offset = offset


class _Tokenizer:
    mapping = {0: "▁hello", 1: "▁world"}

    def ids_to_tokens(self, ids):
        return [self.mapping[i] for i in ids]

    def ids_to_text(self, ids):
        return "".join(self.mapping[i] for i in ids)


class _Model:
    tokenizer = _Tokenizer()


def test_get_word_timestamps():
    hypos = [_Hypo([0, 1], [0.0, 1.0], 0.0)]
    words = get_word_timestamps(hypos, _Model(), time_stride=0.1)
    assert [w.word for w in words] == ["hello", "world"]
