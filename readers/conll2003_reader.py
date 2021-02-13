import torch.tensor
from torchtext import data, datasets


class CoNLL2003Reader:
    def __init__(self, base_path: str):
        self.SRC = data.Field(
            pad_token="<pad>",
            init_token='<s>',
            eos_token='</s>',
            unk_token='<unk>'
        )

        self.TRG = data.Field(
            pad_token="<pad>",
            init_token='<s>',
            eos_token='</s>',
            is_target=True,
            unk_token='<unk>'
        )

        self.fields = [('src', self.SRC), ('trg', self.TRG)]

        self.train_ds = self.read_data(f'{base_path}.train.txt')
        self.valid_ds = self.read_data(f'{base_path}.dev.txt')
        self.test_ds = self.read_data(f'{base_path}.test.txt')

        self.SRC.build_vocab(self.train_ds.src, min_freq=10)
        self.TRG.build_vocab(self.train_ds.trg, min_freq=1)

    def read_data(self, path: str):
        with open(path, encoding='utf-8') as f:
            examples = []
            words = []
            labels = []
            for line in f:
                line = line.strip()
                if not line:
                    examples.append(data.Example.fromlist([words, labels], self.fields))
                    words = []
                    labels = []
                else:
                    # official NN I-NP O
                    columns = line.split()
                    words.append(columns[0])
                    labels.append(columns[1])
            return data.Dataset(examples, self.fields)

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
                if c == field.pad_token:
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
