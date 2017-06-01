from data import bAbIGen
import pickle
from collections import namedtuple
import os
import sys
from natsort import natsorted
import yaml
import pandas as pd

from model import MyModel

with open("config.yml") as config_file:
    config = yaml.load(config_file)

config = namedtuple("Config", config.keys())(*config.values())


with open(config.vocab_file, "rb") as file:
    vocab, rev_vocab = pickle.load(file)

m = MyModel(config)
m.compile()
m.load(sys.argv[2])
gen_wrap = m.generator_wrapper

stats = []
files = []

for file in natsorted(os.listdir(sys.argv[1])):
    test_gen = bAbIGen(os.path.join(sys.argv[1], file), config).generate_batches(config.batch_size, vocab, False)

    score = m.model.evaluate_generator(
        gen_wrap(test_gen),
        10,
    )

    stats.append(score)
    files.append(file)

print(pd.DataFrame(stats, files, ["loss", "elemwise_acc", "answerwise_acc"]))

# print("{}: {:.2} loss, {:.2%} keras acc, {:.2%} accuracy".format(file, score[0], score[1], score[2]))
