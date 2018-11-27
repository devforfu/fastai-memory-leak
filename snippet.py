"""
A snippet demonstrating RAM leakage during model training process.
"""
import argparse

from multiprocessing import cpu_count
import psutil

import fastai
from fastai import defaults, untar_data, URLs
from fastai.vision import ImageDataBunch, create_cnn
from fastai.callbacks import Callback
import torch
from torchvision import models


def main():
    # fastai.show_install()
    bs, n = parse_args()

    print('Creating data and model...')
    path = untar_data(URLs.PETS)
    bunch = ImageDataBunch.from_folder(path, valid='test', bs=bs, num_workers=cpu_count())
    learn = create_cnn(bunch, models.resnet18)

    print('Training model...')
    learn.fit_one_cycle(n, callbacks=[MemoryUsage()])
    print('Done!')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-bs', '--batch-size',
        default=4, type=int,
        help='batch size'
    )
    parser.add_argument(
        '-dev', '--device',
        default='cuda:0',
        help='device to train the model'
    )
    parser.add_argument(
        '-n', '--epochs',
        default=1, type=int,
        help='number of training epochs'
    )
    args = parser.parse_args()
    defaults.device = torch.device(args.device)
    return args.batch_size, args.epochs


class MemoryUsage(Callback):

    def __init__(self, memory_log: str='memory.csv'):
        self.memory_log = memory_log
        self.iter = None
        self._stream = None

    def on_train_begin(self, **kwargs):
        self.iter = 0
        if self._stream is not None:
            self.close()
        self._stream = open(self.memory_log, 'w')
        self._stream.write('index,mem_percent,mem_free,mem_available,mem_used\n')
        super().on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        self.close()

    def on_batch_end(self, **kwargs):
        self.iter += 1
        mem = psutil.virtual_memory()
        record = [self.iter, mem.percent, mem.free, mem.available, mem.used]
        string = ','.join([str(x) for x in record])
        self._stream.write(string + '\n')
        return False

    def close(self):
        self._stream.flush()
        self._stream.close()


if __name__ == '__main__':
    main()
