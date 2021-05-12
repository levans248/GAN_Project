from trainer import Trainer
from GAN_Shakespeare import GAN
from DataLoader import DataLoader
from DataHandler import DataHandler
from utils import create_directories
from sklearn import preprocessing

import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Open shakespeare text file and read in data as `text`
with open('Shakespeare/shakespeare.txt') as f:
    text = f.read()

# We create two dictionaries:
# 1. int2char, which maps integers to characters
# 2. char2int, which maps characters to integers
chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}

#create by running python data_split.py
train = np.loadtxt('train.txt', delimiter="\n", dtype="str")
train_data = []
for row in train:
    encode_string = ''
    for ch in row.strip():
        encode_string += str(char2int[ch])

    if(encode_string != ''):
        train_data.append([int(encode_string), 1])
train_data = np.array(train_data)

test = np.loadtxt('test.txt', delimiter="\n", dtype="str")
test_data = []
for row in test:
    encode_string = ''
    for ch in row.strip():
        encode_string += str(char2int[ch])

    if (encode_string != ''):
        test_data.append([int(encode_string), 1])
test_data = np.array(test_data)

train_loader = DataLoader()
train_data, act_train_labels = train_loader.time_series_to_section(train_data,
                                                                   1,
                                                                   sliding_window_size=200,
                                                                   step_size_of_sliding_window=10)


test_data, act_test_labels = train_loader.time_series_to_section(test_data,
                                                                 1,
                                                                 sliding_window_size=200,
                                                                 step_size_of_sliding_window=10)

print("---Data is successfully loaded")
handler = DataHandler(train_data, test_data)
norm_train = handler.normalise("train", True)
norm_test = handler.normalise("test", True)

print("--- Shape of Training Data:", train_data.shape)
print("--- Shape of Test Data:", test_data.shape)

expt_name = "GAN_Shakespeare_Results"

create_directories(expt_name)
print("start GAN")
gan_ = GAN(norm_train.shape)
print("start trainer")
trainer_ = Trainer(gan_, expt_name)
trainer_.train_gan(epochs=200, batch_size=128, sample_interval=10, train_data=norm_train)

# create_directories(expt_name)
# print("start GAN")
# gan_ = GAN(norm_train.shape)
# print("start trainer")
# trainer_ = Trainer(gan_, expt_name)
# trainer_.train_gan(epochs=200, batch_size=128, sample_interval=10, train_data=norm_train)
