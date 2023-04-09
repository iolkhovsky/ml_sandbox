import argparse
import datetime
import itertools
from math import ceil
import os
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils import read_yaml, get_available_device
from models import build_model
from datasets import build_dataloaders


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join('nmt', 'config', 'train.yaml'),
                        help='Path to training config')
    args = parser.parse_args()
    return args


def train(
        device, model, optimizer, train_dataloader, en_tokenizer, ru_tokenizer,
        val_dataloader=None, epochs=0, iterations=100, logs_path='logs',
        checkpoints_path='checkpoints', val_steps=50,
):
    model.to(device)

    total_steps = iterations
    if epochs:
        total_steps = len(train_dataloader) * epochs
    else:
        epochs = ceil(iterations / len(train_dataloader))
    
    val_iter = None
    if val_dataloader is not None:
        val_iter = itertools.cycle(iter(val_dataloader))

    session_timestamp = str(datetime.datetime.now())
    session_timestamp = session_timestamp.replace(" ", "").replace(":", "-").replace(".", "-")
    logs_path = os.path.join(
        logs_path,
        session_timestamp,
    )
    os.makedirs(logs_path)

    writer = SummaryWriter(log_dir=logs_path)

    step = 0
    with tqdm(total=total_steps) as pbar:
        for epoch in range(epochs):
            for train_batch in train_dataloader:
                model.train()
                optimizer.zero_grad()

                in_tokens = torch.Tensor(train_batch['encoder_input']).long().to(device)
                dec_inputs = torch.Tensor(train_batch['decoder_input']).long().to(device)
                dec_targets = torch.Tensor(train_batch['decoder_output']).long().to(device)

                loss = model(
                    tokens=in_tokens,
                    dec_input=dec_inputs,
                    dec_target=dec_targets,
                )
                
                loss.backward()
                optimizer.step()
                
                if val_steps and (step % val_steps == 0) and (val_iter is not None):
                    val_batch = next(val_iter)
                    in_tokens = torch.Tensor(val_batch['encoder_input']).long().to(device)
                    dec_targets = torch.Tensor(val_batch['decoder_output']).long().to(device)

                    in_tokens = in_tokens[0].unsqueeze(0)
                    dec_targets = dec_targets[0].unsqueeze(0)

                    model.eval()
                    pred = model(in_tokens)

                    if in_tokens.device != torch.device('cpu'):
                        in_tokens = in_tokens.cpu()
                        dec_targets = dec_targets.cpu()
                        pred = [x.cpu() for x in pred]
                    
                    input_sample = en_tokenizer.decode_line(in_tokens[0])
                    target_sample = ru_tokenizer.decode_line(dec_targets[0])
                    predicted_sample = ru_tokenizer.decode_line(pred)
                    writer.add_text('Validation/Input', input_sample, global_step=step)
                    writer.add_text('Validation/Target', target_sample, global_step=step)
                    writer.add_text('Validation/Prediction', predicted_sample, global_step=step)
                
                pbar.update(1)
                if device != torch.device('cpu'):
                    loss = loss.cpu()
                loss = loss.detach().item()
                pbar.set_description(f'Step: {step} Epoch: {epoch} Loss: {loss}')
                writer.add_scalar('Loss/Train', loss, step)
                step += 1

                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    print('Completed')


def run(args):
    training_config = read_yaml(args.config)

    device = training_config['device']
    if device == 'auto':
        device = get_available_device()
    device = torch.device(device)

    model, optimizer = build_model(
        training_config['model'],
        training_config['optimizer'],
    )
    print(model)

    train_dataloader, val_dataloader, en_tokenizer, ru_tokenizer = \
        build_dataloaders(training_config['dataset'])

    train(
        device=device,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        en_tokenizer=en_tokenizer,
        ru_tokenizer=ru_tokenizer,
        val_dataloader=val_dataloader,
        epochs=training_config['epochs'],
        iterations=training_config['iterations'],
        logs_path=training_config['logs'],
        checkpoints_path=training_config['checkpoints'],
        val_steps=training_config['validation_steps'],
    )


if __name__ == '__main__':
    run(parse_cmd_args())
