'''
From https://github.com/IST-DASLab/gptq/blob/main/datautils.py
'''

import numpy as np
import torch
from functools import lru_cache


@lru_cache
def get_gsm8k(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random

    # load gsm8k dataset
    traindata = load_dataset('gsm8k', 'main', split='train')
    testdata = load_dataset('gsm8k', 'main', split='test')

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # set seed
    random.seed(seed)

    # prepare dataset for training
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        encoded = tokenizer(traindata[i]['question'], return_tensors='pt', padding="max_length", truncation=True, max_length=seqlen)
        inp = encoded.input_ids
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # prepare dataset for test
    test_tokens = []
    for i in range(len(testdata)):
        encoded = tokenizer(testdata[i]['question'], return_tensors='pt', padding="max_length", truncation=True, max_length=seqlen)
        test_tokens.append(encoded.input_ids)

    # converting the list of tensors into a single tensor and wrapping it in a dictionary
    test_tokens_tensor = torch.cat(test_tokens, dim=0)
    test_data_dict = {'input_ids': test_tokens_tensor}

    return trainloader, test_data_dict





@lru_cache
def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'gsm8k' in name:
        return get_gsm8k(nsamples, seed, seqlen, model)


@lru_cache
def get_test_tokens(
    name, seed=0, seqlen=2048, model=''
):
    train_samples = 0
    if name == 'gsm8k':
        return get_gsm8k(train_samples, seed, seqlen, model)[1]['input_ids']

