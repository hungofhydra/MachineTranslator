#Necessary Library  
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
from turtle import shape
from TextProcessing import text_processing
import numpy as np
import typing
from typing import Any, Tuple
import tensorflow as tf
import tensorflow_text as tf_text
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pathlib
from ShapeChecker import ShapeChecker
from Encoder import Encoder
from Attention import Attention

reloaded = tf.saved_model.load('modelEnglishToVietnamese')

while True:
    val = input("Enter your value: ")
    testText = tf.constant([val])


    result = reloaded.tf_translate(testText)
    for tr in result['text']:
        print(tr.numpy().decode())

    
    









