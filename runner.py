import torch
from critics import BasicCritic
from decoders import DenseDecoder
from encoders import DenseEncoder
from loader import DataLoader
from models import SteganoGAN
from utils import text_to_bits, bits_to_bytearray, bytearray_to_text
from collections import Counter
import numpy as np
import os
from imageio import imread, imwrite
import matplotlib.pyplot as plt
import imageio
from imageio import imread, imwrite

# Load the data
train = DataLoader('\imagedataset\cba')
validation = DataLoader('\imagedataset\dba')
# Create the SteganoGAN instance
steganogan = SteganoGAN(1, DenseEncoder, DenseDecoder, BasicCritic, hidden_size=32, cuda=True, verbose=True)
steganogan.fit(train, validation, epochs=1)
steganogan.save('C:\Research\dense.steg')
steganogan = SteganoGAN.load('C:\Research\dense', cuda=False)
steganogan.encode('lenna.png', 'output.png', 'changwon')
steganogan.decode('output.png')