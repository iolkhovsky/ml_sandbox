import os
from itertools import cycle
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Tokenizer, OpusTranslationDataset, KaggleTranslationDataset
from model import Translator
from utils import get_available_device


DATA_ROOT = 'nmt/data'
# TRAIN_RU = os.path.join(DATA_ROOT, 'opus/train_ru.txt')
# TRAIN_EN = os.path.join(DATA_ROOT, 'opus/train_en.txt')
# VAL_RU = os.path.join(DATA_ROOT, 'opus/test_ru.txt')
# VAL_EN = os.path.join(DATA_ROOT, 'opus/test_en.txt')
PATH =  os.path.join(DATA_ROOT, 'kaggle_rus_dict/rus.txt')
TRAIN_BATCH_SIZE = 32
MAX_WORDS = 10

en_tokenizer = Tokenizer()
ru_tokenizer = Tokenizer()

dataset = KaggleTranslationDataset(
    path=PATH,
    en_tokenizer=en_tokenizer,
    ru_tokenizer=ru_tokenizer,
    fit_tokenizer=True,
    max_words=MAX_WORDS,
)
dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
print(f'Dataset size: {len(dataset)}')

# print('Loading training dataset...')
# train_dataset = OpusTranslationDataset(
#     en_data_path=TRAIN_EN,
#     ru_data_path=TRAIN_RU,
#     en_tokenizer=en_tokenizer,
#     ru_tokenizer=ru_tokenizer,
#     fit_tokenizer=True,
#     max_words=40
# )
# train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# print(f'Training dataset size: {len(train_dataset)}')

# print('Loading validation dataset...')
# val_dataset = OpusTranslationDataset(
#     en_data_path=VAL_EN,
#     ru_data_path=VAL_RU,
#     en_tokenizer=en_tokenizer,
#     ru_tokenizer=ru_tokenizer,
#     max_words=40
# )
# val_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
# print(f'Validation dataset size: {len(val_dataset)}')

device = 'cpu' # device = get_available_device()
model = Translator(max_length=MAX_WORDS)
model.to(device)

# train_iter = iter(train_dataloader)
# val_iter = iter(val_dataloader)
train_iter = cycle(iter(dataloader))
optimizer = optim.RMSprop(model.parameters(), lr=0.01)
# optimizer = optim.SGD(model.parameters(), lr=1e-2)

steps_to_train = 10000

print('Running training loop...')
with tqdm(total=steps_to_train) as pbar:
    for step in range(steps_to_train):
        # try:
        if True:
            model.train()
            optimizer.zero_grad()
            
            sample = next(train_iter)
            in_tokens = torch.Tensor(sample['encoder_input']).long().to(device)
            dec_inputs = torch.Tensor(sample['decoder_input']).long().to(device)
            dec_targets = torch.Tensor(sample['decoder_output']).long().to(device)

            loss = model(
                tokens=in_tokens,
                dec_input=dec_inputs,
                dec_target=dec_targets,
            )
            
            loss.backward()
            optimizer.step()
            
            if step % 50 == 0:
                val_sample = next(train_iter)
                in_tokens = torch.Tensor(val_sample['encoder_input']).long().to(device)
                dec_targets = torch.Tensor(val_sample['decoder_output']).long().to(device)

                in_tokens = in_tokens[0].unsqueeze(0)
                dec_targets = dec_targets[0].unsqueeze(0)

                model.eval()
                pred = model(in_tokens)

                print('\n\nEvaluation')
                if in_tokens.device != torch.device('cpu'):
                    in_tokens = in_tokens.cpu()
                    dec_targets = dec_targets.cpu()
                    pred = [x.cpu() for x in pred]
                print(f'Input sequence:\n{en_tokenizer.decode_line(in_tokens[0])}')
                print(f'Target sequence:\n{ru_tokenizer.decode_line(dec_targets[0])}')
                print(f'Predicted sequence:\n{ru_tokenizer.decode_line(pred)}')

            pbar.set_description(f'Step {step} Loss: {loss.detach()}')
        # except Exception as e:
        #     print(f'Got exception: {e}')
        pbar.update(1)
