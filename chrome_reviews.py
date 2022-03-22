import pandas as pd
import torch
import random
import numpy
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import warnings
warnings.filterwarnings('ignore')

def predictionsfn(file):
    df = pd.read_csv(file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = df[['Text', 'Star']]
    df.dropna(inplace=True)
    reviews = df.Text.values
    labels = df.Star.values
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    max_length = 64
    input_ids = []
    attention_masks = []
    for rev in reviews:
        encoded_dict = tokenizer.encode_plus(
            rev,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    dataset = TensorDataset(input_ids, attention_masks)
    batch_size = 32

    test_dataloader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size
    )
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )

    model.load_state_dict(torch.load("model.pt",map_location=torch.device('cpu')))
    predictions = []
    model.eval()

    for batch in test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)

        with torch.no_grad():
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           return_dict=True)

        loss = result.loss
        logits = result.logits
        classes = torch.argmax(logits, dim=1)
        classes=classes.numpy()
        logits = logits.detach().cpu().numpy()
        predictions.append(classes)
    flat_pred = [item for sublist in predictions for item in sublist]
    df['pred'] = flat_pred
    df = df[(df['pred'] == 1) & ((df['star']==1) | (df['star']==2))]
    return df

import streamlit as st

import os


st.title('Review discrepancy with rating')

uploaded_file = st.file_uploader("Upload a csv or xlsx file",type=['csv','xlsx'])
if uploaded_file is not None:
    if st.button("Process"):
            file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(file_details)

            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head())
            uploaded_file.seek(0)
            prediction = predictionsfn(uploaded_file)
            st.text('Predictions :-')
            st.dataframe(prediction)

