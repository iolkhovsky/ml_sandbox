import os
from itertools import cycle
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import Tokenizer, OpusTranslationDataset, KaggleTranslationDataset
from models import BasicLstmTranslator
from utils import get_available_device


DATA_ROOT = 'nmt/data'
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

device = 'cpu' # device = get_available_device()
model = BasicLstmTranslator(max_length=MAX_WORDS)
model.to(device)

train_iter = cycle(iter(dataloader))
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

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
        pbar.update(1)
