#Necessary Library  
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pathlib



def load_data():
    input_texts=[]
    target_texts=[]
    
    #read dataset file
    with open('Dataset/eng-vie.txt','r',encoding='utf-8') as f:
        rows=f.read().split('\n')
    
        
    for row in rows:
        input_text,target_text,contribution_text = row.split('\t')

        input_texts.append(input_text.lower())
        target_texts.append(target_text.lower())

    return input_texts, target_texts

input_texts, target_texts =  load_data()
