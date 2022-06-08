from tkinter import *
from tkinter import Label, Tk, W, E
from tkinter.ttk import Frame, Button, Style
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

class Example(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.initUI()

    def initUI(self):

        self.parent.title("Translator")

        Style().configure("TButton", padding=(0, 5, 0, 5), font='serif 10')

        self.columnconfigure(0, pad=3)
        self.columnconfigure(1, pad=3)
        self.columnconfigure(2, pad=3)

        self.rowconfigure(0, pad=3)
        self.rowconfigure(1, pad=3)
        self.rowconfigure(2, pad=3)
        self.rowconfigure(3, pad=3)
        self.rowconfigure(4, pad=3)

        def get_input():
            result = inputtxt.get("1.0", "end")
            return result

        label1 = Label(self, text="Origin")
        label1.grid(row=1, column=0)
        inputtxt = Text(self, height=5,
                        width=25,
                        bg="light yellow")
        inputtxt.grid(row=2, column=0)

        label2 = Label(self, text="Translated")
        label2.grid(row=1, column=2)
        outputtxt = Text(self, height=5,
                         width=25,
                         bg="light blue")
        outputtxt.grid(row=2, column=2)

        def EngToVie():
            outputtxt.configure(state="normal")
            outputtxt.delete("1.0","end")
            input = get_input()
            inputlanguage = tf.constant([input])
            result = reloadedEngtoVie.tf_translate(inputlanguage)
            for tr in result['text']:
                translated = tr.numpy().decode()
            outputtxt.insert(INSERT, translated)
            outputtxt.configure(state="disabled")

        def VieToEng():
            outputtxt.configure(state="normal")
            outputtxt.delete("1.0","end")
            input = get_input()
            inputlanguage = tf.constant([input])
            result = reloadedVietoEng.tf_translate(inputlanguage)
            for tr in result['text']:
                translated = tr.numpy().decode()
            outputtxt.insert(INSERT, translated)
            outputtxt.configure(state="disabled")

        def clear():
            outputtxt.configure(state="normal")
            inputtxt.delete("1.0","end")
            outputtxt.delete("1.0","end")
            outputtxt.configure(state="disabled")

        bt1 = Button(self, text="--EngToVie->", command=EngToVie)
        bt1.grid(row=3, column=1)

        bt2 = Button(self, text="--VieToEng->", command=VieToEng)
        bt2.grid(row=4, column=1)

        bt3 = Button(self, text="Clear all", command=clear)
        bt3.grid(row=5, column=1)

        self.pack()

        outputtxt.configure(state="disabled")

reloadedEngtoVie = tf.saved_model.load('modelEnglishToVietnamese')
reloadedVietoEng = tf.saved_model.load('modelVietnameseToEnglish')
root = Tk()
app = Example(root)
root.mainloop()

