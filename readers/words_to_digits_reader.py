import torch.tensor
from torchtext import data, datasets


class Words2DigitsReader:
    def __init__(self, base_path: str, ext_src: str = '.lemmas', ext_trg: str = '.digits'):
        self.SRC = data.Field(
            pad_token="<pad>",
            fix_length=29,
            init_token='<s>',
            eos_token='</s>',
            tokenize=lambda x: x.split()[::-1],
            unk_token=None
        )

        self.TRG = data.Field(
            pad_token="<pad>",
            fix_length=18,
            init_token='<s>',
            eos_token='</s>',
            tokenize=lambda x: x.split(),
            is_target=True,
            unk_token=None
        )

        self.train_ds = datasets.TranslationDataset(
                path=f'{base_path}.train',
                exts=(ext_src, ext_trg),
                fields=(self.SRC, self.TRG)
            )

        self.valid_ds = datasets.TranslationDataset(
                path=f'{base_path}.valid',
                exts=(ext_src, ext_trg),
                fields=(self.SRC, self.TRG)
            )

        self.SRC.build_vocab(self.train_ds.src, min_freq=1)
        self.TRG.build_vocab(self.train_ds.trg, min_freq=1)

    @staticmethod
    def _encode(field, x):
        if isinstance(x, str):
            x = [x]
        return field.process(list(map(field.preprocess, x)))

    @staticmethod
    def _decode(field, x: torch.LongTensor):
        assert len(x.shape) == 2
        max_len, batch_size = x.shape
        res = []
        for i in range(batch_size):
            res_i = []
            for j in range(max_len):
                c = field.vocab.itos[x[j][i]]
                res_i.append(c)
                if c == field.eos_token:
                    break
            res.append(res_i)
        return res

    def encode_src(self, x):
        return self._encode(self.SRC, x)

    def decode_src(self, x: torch.LongTensor):
        return self._decode(self.SRC, x)

    def encode_trg(self, x):
        return self._encode(self.TRG, x)

    def decode_trg(self, x: torch.LongTensor):
        return self._decode(self.TRG, x)
