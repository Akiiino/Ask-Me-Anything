from data import bAbIGen
import pickle
from collections import namedtuple
import keras

from model import MyModel

config = {
    "max_context_len": 500,
    "max_sentence_num": 100,
    "max_sentence_len": 15,
    "max_question_len": 15,
    "max_answer_len": 6,
    "vocab_size": 200,
    "hidden_dim": 80
}

config = namedtuple("Config", config.keys())(*config.values())


with open("vocab.pickle", "rb") as file:
    vocab, rev_vocab = pickle.load(file)

train_gen = bAbIGen("../mixed/train_all.txt", config).generate_batches(1000, vocab)
valid_gen = bAbIGen("../mixed/valid_all.txt", config).generate_batches(1000, vocab)


m = MyModel(config)
m.compile()
gen_wrap = m.generator_wrapper

m.model.fit_generator(
    gen_wrap(train_gen),
    100,
    100,
    validation_data=gen_wrap(valid_gen),
    validation_steps=20,
    callbacks=[
        keras.callbacks.ModelCheckpoint("best_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(patience=20)
    ]
)
