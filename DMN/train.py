from code.data import bAbIGen
from code.model import MyModel
import pickle
from collections import namedtuple
import keras
import sys
import os
import yaml


with open("config.yml") as file:
    config = yaml.load(file)

config = namedtuple("Config", config.keys())(*config.values())

with open(config.vocab_file, "rb") as file:
    vocab, rev_vocab = pickle.load(file)

train_gen = bAbIGen(
    os.path.join(config.bAbI_folder, "train_all.txt"),
    config
).generate_batches(config.batch_size, vocab)
valid_gen = bAbIGen(
    os.path.join(config.bAbI_folder, "valid_all.txt"),
    config
).generate_batches(config.batch_size, vocab)


m = MyModel(config)
if len(sys.argv) > 1:
    m.load(sys.argv[1])

gen_wrap = m.generator_wrapper

m.model.fit_generator(
    gen_wrap(train_gen),
    100,
    500,
    validation_data=gen_wrap(valid_gen),
    validation_steps=20,
    callbacks=[
        keras.callbacks.ModelCheckpoint("/checkpoint_t", save_best_only=True),
        keras.callbacks.EarlyStopping(patience=20)
    ]
)
