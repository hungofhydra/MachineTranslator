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


#Load Dataset into array
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


#Turn text to Unicode character
def tf_lower_and_split_punct(text):


  text = tf_text.normalize_utf8(text, 'NFKD')
  text = tf.strings.lower(text)
  text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
  text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')

  text = tf.strings.strip(text)

  text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
  return text


def text_processing():
    input_texts, target_texts =  load_data()
    BUFFER_SIZE = len(input_texts)
    BATCH_SIZE = 64

    dataset = tf.data.Dataset.from_tensor_slices((input_texts, target_texts)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)

    for example_input_batch, example_target_batch in dataset.take(1):
      print(example_input_batch[:1])
      print()
      print(example_target_batch[:1])
      print('===========================================')
      break

    max_vocab_size = 5000

    #Change text to Vector
    #English
    input_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size)
    input_text_processor.adapt(input_texts)


    #Vietnamese
    output_text_processor = tf.keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct,
        max_tokens=max_vocab_size)

    output_text_processor.adapt(target_texts)

    example_tokens = output_text_processor(example_target_batch)


    input_vocab = np.array(output_text_processor.get_vocabulary())
    tokens = input_vocab[example_tokens[0].numpy()]
    ' '.join(tokens)
    print(tokens)


