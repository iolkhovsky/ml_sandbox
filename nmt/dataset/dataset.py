import numpy as np
from torch.utils.data import Dataset


class OpusTranslationDataset(Dataset):
    def __init__(
        self,
        en_data_path,
        ru_data_path,
        en_tokenizer,
        ru_tokenizer,
        fit_tokenizer=False,
        max_words=40
    ):
        self.en_lang = en_tokenizer
        self.ru_lang = ru_tokenizer
        self._max_words = max_words
        
        with open(en_data_path, 'rt', encoding='utf-8') as f:
            self._inputs = f.readlines()
        with open(ru_data_path, 'rt', encoding='utf-8') as f:
            self._targets = f.readlines()
       
        if fit_tokenizer:
            self.en_lang.fit(self._inputs)
            self.ru_lang.fit(self._targets)   

        self._filter_samples()
        assert len(self._inputs) == len(self._targets)

    def _filter_samples(self):
        valid_indices = []
        
        def valid(input_line, target_line):
            input_tokens = self.en_lang.encode_line(input_line)
            target_tokens = self.ru_lang.encode_line(target_line)
            
            input_unk_share = input_tokens.count(self.en_lang.unk_token) / len(input_tokens)
            target_unk_share = target_tokens.count(self.ru_lang.unk_token) / len(target_tokens)

            return (self._max_words * 0.8 <= len(input_tokens) <= self._max_words - 2) and \
                (self._max_words * 0.8 <= len(target_tokens) <= self._max_words - 2) and \
                (input_unk_share <= 0.2) and \
                (target_unk_share <= 0.2)

        for i in range(len(self._inputs)):
            if valid(self._inputs[i], self._targets[i]):
                valid_indices.append(i)
        
        self._inputs = [self._inputs[x] for x in valid_indices]
        self._targets = [self._targets[x] for x in valid_indices]

    def __len__(self):
        return len(self._inputs)
    
    @staticmethod
    def _pad(seq, tokenizer, size, prepadding=False):
        pad_length = size - len(seq)
        if pad_length == 0:
            return seq
        if prepadding:
            return pad_length * [tokenizer.pad_token] + seq
        else:
            return seq + pad_length * [tokenizer.pad_token]

    def __getitem__(self, index):       
        in_sentence = self._inputs[index]
        encoder_input = self.en_lang.encode_line(in_sentence)[:self._max_words]
        encoder_input = OpusTranslationDataset._pad(encoder_input, self.en_lang, self._max_words, prepadding=True)

        target_sentence = self._targets[index]
        decoder_output = self.ru_lang.encode_line(target_sentence)[:self._max_words] + [self.ru_lang.stop_token]
        decoder_input = [self.ru_lang.start_token] + decoder_output
        decoder_output = OpusTranslationDataset._pad(decoder_output, self.ru_lang, self._max_words)
        decoder_input = OpusTranslationDataset._pad(decoder_input, self.ru_lang, self._max_words)
        
        return {
            'encoder_input': np.asarray(encoder_input),
            'decoder_input': np.asarray(decoder_input),
            'decoder_output': np.asarray(decoder_output),
        }


class KaggleTranslationDataset(Dataset):
    def __init__(
        self,
        path,
        en_tokenizer,
        ru_tokenizer,
        fit_tokenizer=False,
        max_words=10
    ):
        self.en_lang = en_tokenizer
        self.ru_lang = ru_tokenizer
        self._max_words = max_words
        
        with open(path, 'rt', encoding='utf-8') as f:
            self._samples = f.readlines()
            self._inputs = [x.split('\t')[0] for x in self._samples]
            self._targets = [x.split('\t')[1] for x in self._samples]
       
        if fit_tokenizer:
            self.en_lang.fit(self._inputs)
            self.ru_lang.fit(self._targets)   

        assert len(self._inputs) == len(self._targets)

    def __len__(self):
        return len(self._inputs)
    
    @staticmethod
    def _pad(seq, tokenizer, size, prepadding=False):
        pad_length = size - len(seq)
        if pad_length == 0:
            return seq
        if prepadding:
            return pad_length * [tokenizer.pad_token] + seq
        else:
            return seq + pad_length * [tokenizer.pad_token]

    def __getitem__(self, index):       
        in_sentence = self._inputs[index]
        encoder_input = self.en_lang.encode_line(in_sentence)[:self._max_words]
        encoder_input = KaggleTranslationDataset._pad(encoder_input, self.en_lang, self._max_words, prepadding=True)

        target_sentence = self._targets[index]
        decoder_output = self.ru_lang.encode_line(target_sentence) + [self.ru_lang.stop_token]
        decoder_output = decoder_output[:self._max_words]
        decoder_input = [self.ru_lang.start_token] + decoder_output
        decoder_input = decoder_input[:self._max_words]
        decoder_output = KaggleTranslationDataset._pad(decoder_output, self.ru_lang, self._max_words)
        decoder_input = KaggleTranslationDataset._pad(decoder_input, self.ru_lang, self._max_words)

        assert len(encoder_input) == self._max_words
        assert len(decoder_input) == self._max_words
        assert len(decoder_output) == self._max_words
        
        return {
            'encoder_input': np.asarray(encoder_input),
            'decoder_input': np.asarray(decoder_input),
            'decoder_output': np.asarray(decoder_output),
        }
