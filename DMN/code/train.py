from data import bAbIGen
import pickle
from collections import namedtuple
import keras
import sys
import os
import yaml

from model import MyModel

config = yaml.load("config.yml")

config = namedtuple("Config", config.keys())(*config.values())

with open(config.vocab_file, "rb") as file:
    vocab, rev_vocab = pickle.load(file)

train_gen = bAbIGen(
    os.path.join(config.bAbI_folder, "train_all.txt"),
    config
).generate_batches(config.batch_size, vocab, False)
valid_gen = bAbIGen(
    os.path.join(config.bAbI_folder, "valid_all.txt"),
    config
).generate_batches(config.batch_size, vocab, False)


m = MyModel(config)
m.compile()

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
        keras.callbacks.ModelCheckpoint("/output/best_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(patience=20)
    ]
)
