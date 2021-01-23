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

        # self.valid_ds = datasets.TranslationDataset(
        #         path=f'{base_path}.test',
        #         exts=(ext_src, ext_trg),
        #         fields=(self.SRC, self.TRG)
        #     )

        self.SRC.build_vocab(self.train_ds.src, min_freq=1)
        self.TRG.build_vocab(self.train_ds.trg, min_freq=1)

    def encode_src(self, x):
        if isinstance(x, str):
            x = [x]
        return self.SRC.process(list(map(self.SRC.preprocess, x)))

    def decode_src(self, x: torch.LongTensor):
        assert len(x.shape) == 2
        x = x.T
        batch_size, trg_len = x.shape
        res = []
        for i in range(batch_size):
            res.append(
                list(map(lambda t: self.SRC.vocab.itos[t], x[i]))
            )
        return res

    def encode_trg(self, x):
        if isinstance(x, str):
            x = [x]
        return self.TRG.process(list(map(self.TRG.preprocess, x)))

    def decode_trg(self, x: torch.LongTensor):
        assert len(x.shape) == 2
        x = x.T
        batch_size, trg_len = x.shape
        res = []
        for i in range(batch_size):
            res_i = []
            for j in range(trg_len):
                c = self.TRG.vocab.itos[x[i][j]]
                res_i.append(c)
                if c == self.TRG.eos_token:
                    break
            res.append(res_i)
        return res
