from torchtext import data
import spacy
import re
import os
import pandas as pd
from torchtext.data import TabularDataset, Iterator, BucketIterator
import numpy as np
import torch
from torch.autograd import Variable

def readData(source_path, target_path):
    source_data = open(source_path).read().strip().split("\n")
    target_data = open(target_path).read().strip().split("\n")
    return source_data, target_data

class tokenize():
    def __init__(self,language):
        self.nlp = spacy.load(language)

    def tokenizer(self, sentence):
        nlp = spacy.load(source_language)
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]


def processingData(source_data, target_data, source_language, target_language):
    max_length = 256
    DATA = {
        "source":[line for line in source_data],
        "target":[line for line in target_data]
    }
    df = pd.DataFrame(DATA,columns=['source','target'])
    df.to_csv("translate_transformer_temp.csv", index=False)
    tokenize = lambda x: x.split(' ')
    source_filed = data.Field(lower=True, tokenize= tokenize)
    target_filed = data.Field(lower=True, tokenize=tokenize,init_token='<sos>', eos_token= '<eos')
    data_fileds = [("source",source_filed),("target",target_filed)]
    source_filed.build_vocab(source_data)
    target_filed.build_vocab(target_data)
    DATA = TabularDataset("translate_transformer_temp.csv",format='csv', fields=data_fileds)
    os.remove("translate_transformer_temp.csv")
    return DATA,source_filed,target_filed

# Tạo mặt nạ cho câu

def nopeakMask(size):          #size: Kích thước của câu
    np_mask = np.triu(np.ones((1, size, size)),k=1).astype("uint8")
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    return np_mask

def createMasks(source_data, target_data, source_padding, target_padding):
    source_mask = (source_data != source_padding).unsqueeze(-2)

    if target_data != None:
        target_mask = (target_data != target_padding).unsqueeze(-2)
        size = target_data.size(1)
        np_mask = nopeakMask(size)
        target_mask = target_mask & np_mask
    else:
        target_mask = None
    return source_mask , target_mask



