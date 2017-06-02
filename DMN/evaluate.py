from code.data import bAbIGen
from code.model import MyModel
import pickle
from collections import namedtuple
import os
import sys
from natsort import natsorted
import yaml
import pandas as pd


with open("config.yml") as config_file:
    config = yaml.load(config_file)

config = namedtuple("Config", config.keys())(*config.values())

with open(config.vocab_file, "rb") as file:
    vocab, rev_vocab = pickle.load(file)

m = MyModel(config)
m.load(sys.argv[2])
gen_wrap = m.generator_wrapper

stats = []
files = []

for file in natsorted(os.listdir(config.babi_folder)):
    test_gen = bAbIGen(os.path.join(config.babi_folder, file), config).generate_batches(config.batch_size, vocab, False)

    score = m.model.evaluate_generator(
        gen_wrap(test_gen),
        1,
    )

    stats.append(score)
    files.append(file)

print(pd.DataFrame(stats, files, ["loss", "elemwise_acc", "answerwise_acc"]))
