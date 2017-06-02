import pickle
from collections import namedtuple
import sys
import yaml

from code.model import MyModel

with open("config.yml") as file:
    config = yaml.load(file)

config = namedtuple("Config", config.keys())(*config.values())

with open(config.vocab_file, "rb") as file:
    vocab, rev_vocab = pickle.load(file)

m = MyModel(config)
m.load(sys.argv[1])

print("Enter your question:")
question = sys.stdin.read()

print(m.predict_from_text(question, vocab, rev_vocab))
