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


class Words2DigitsIterator(data.Iterator):
    def __init__(self, *args, **kwargs):
        super(Words2DigitsIterator, self).__init__(*args, **kwargs)
        self.batches = []

    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn
                    )
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


class Words2DigitBatch(data.Batch):
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        super(Words2DigitBatch, self).__init__()
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
