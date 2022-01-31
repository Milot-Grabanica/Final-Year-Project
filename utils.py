#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import copy
import numpy as np
import transformers
import tweepy
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import warnings

warnings.filterwarnings("ignore")

# This code will check if we have gpu then code will run on gpu else it will run on cpu.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#### Model Parameters
MAX_LENGTH = 150
models_path = './models/'
model_name = 'Bert_Model_HS_2.pth'
PATH = models_path + model_name
# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# Pre Processing
def preprocess(tweet):
    pattern_for_space = '\s+'
    pattern_giant_url = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                         '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    pattern_mention = '@[\w\-]+'
    tweet = re.sub(pattern_for_space, ' ', tweet)  # removing spaces
    tweet = re.sub(pattern_giant_url, '', tweet)  # removing URL
    tweet = re.sub(pattern_mention, '', tweet)  # removing mentioning
    tweet = re.sub("[^a-zA-Z0-9']+", " ", tweet)  # removing other words
    tweet = tweet.strip()  # remove first and last space

    return tweet


# ### Model Defination
class BERT_Model(nn.Module):
    def __init__(self):
        super(BERT_Model, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(
            'bert-base-uncased')  # Use the 12-layer BERT model, with an uncased vocab
        self.out1 = nn.Linear(768, 2)
        # self.out2 = nn.Linear(500, 128)
        # self.out3 = nn.Linear(128, 2)
        self.drop_out = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.drop_out(output_1)
        x = self.out1(x)
        # x = self.drop_out(x)
        # x = self.relu(x)
        # x = self.out2(x)
        # x = self.drop_out(x)
        # x = self.relu(x)
        # x = self.out3(x)
        return x

##### Load Pretrained Model
def load_model():
    model = torch.load(PATH, map_location=device)
    return model

### Convert tweet in encoding 
def get_encoding(tweet):
    encoding = tokenizer.encode_plus(
        tweet,  # txt to encode.
        None,  # text_pair Optional second sequence to be encoded
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LENGTH,  # Truncate all if longer than max len.
        pad_to_max_length=True,  # pad sentences shorter than max len
        truncation=True
    )
    return encoding

##### Get the prediction given that model and encoding
def get_prediction(encoding, model):
    ids = torch.tensor(encoding['input_ids'])
    token_type_ids = torch.tensor(encoding["token_type_ids"])
    mask = torch.tensor(encoding['attention_mask'])
    # loading the inputs and outputs into device
    ids = torch.unsqueeze(ids, 0).to(device)
    mask = torch.unsqueeze(mask, 0).to(device)

    prediction = model(ids=ids, mask=mask, token_type_ids=None)
    _, predicted = torch.max(prediction, 1)
    return predicted.cpu().detach().numpy()[0]

##### We will get the tweet randomly given that keyword
def get_random_tweet_from_tweepy(searchterm1):
    API_Key = 'b3IS6sWt8WfYuA4KRH2Qo2s8d'
    API_Key_Secret = 'TcY43bugzw7L9oox4MSENUBXs52xmpTMtgYNxyeY2V6m3FSt0J'

    consumer_key = API_Key
    consumer_secret = API_Key_Secret

    access_token = "380452810-0d1VKzF7z6VXtAs9oku2MXwvHzIoLmanwlOy1sBg"
    access_token_secret = "3SNT60Ttd8yN4i2y7lgXPUWBbXw5AlQoLxxDKbiP4sPxp"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
    to_return = ''
    public_tweets = api.home_timeline(trim_user=True)
    
    ### search the tweets given the keyword
    search = tweepy.Cursor(api.search, 
                           q= searchterm1,# and searchterm2,
                           lang="en",                    
                           result_type="recent").items(10)

    ### We will select one random tweeet whose length is greater then 35
    for tweet in search:
        toreturn = str(tweet.text)
        if len(toreturn) > 35:
            to_return = toreturn
            if np.random.random() > 0.8 :
                break

    ### remove some extraa information in the tweet and only get the tweet
    to_return = to_return.split(')')[-1]

    return to_return

