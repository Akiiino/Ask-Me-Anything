from data import generate_from_string
import pickle
from collections import namedtuple
import numpy as np
import sys
import yaml

from model import MyModel

with open("config.yml") as file:
    config = yaml.load(file)

config = namedtuple("Config", config.keys())(*config.values())

with open(config.vocab_file, "rb") as file:
    vocab, rev_vocab = pickle.load(file)

m = MyModel(config)
m.load(sys.argv[1])

print("Enter your question:")
q = sys.stdin.read()

print(" ".join(
    [
        word for word in [
            rev_vocab[num] for num in np.argmax(
                m.model.predict(
                    generate_from_string(q, config, vocab)
                ),
                axis=-1)[0]
        ]
        if word != "PAD"
    ]
).replace(" .", ".").replace(" ,", ","))
