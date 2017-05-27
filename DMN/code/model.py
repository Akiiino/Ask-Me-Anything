from keras import backend as K
from keras.layers import (
    Input, Dense, Embedding, Bidirectional, GRU, RepeatVector, concatenate, Lambda, Layer, BatchNormalization
)

import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.optimizers import Adam


class MyEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, **kwargs):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        super(MyEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(self.vocab_size, self.embedding_dim),
            initializer="uniform",
            trainable=True,
            name="weight"
        )
        super(MyEmbedding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape + [self.embedding_dim])

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)


class MyModel:
    def __init__(self, config):
        self.config = config

        question_input = Input((config.max_question_len, ))

        embedded_question = Embedding(config.vocab_size, config.hidden_dim)(question_input)
        summarized_question = Bidirectional(GRU(config.hidden_dim))(embedded_question)
        summarized_question = BatchNormalization()(summarized_question)
        repeated_question = RepeatVector(config.max_sentence_num)(summarized_question)

        context_input = Input((config.max_sentence_num, config.max_sentence_len), dtype='int32')
        embedded_context = MyEmbedding(config.vocab_size, config.hidden_dim)(context_input)
        embedded_context = Lambda(lambda x: K.sum(x, axis=2))(embedded_context)
        embedded_context = BatchNormalization()(embedded_context)

        pre_out = concatenate([repeated_question, embedded_context])
        pre_out = Bidirectional(GRU(config.hidden_dim))(pre_out)
        pre_out = Dense(config.hidden_dim, activation="relu")(pre_out)
        pre_out = BatchNormalization()(pre_out)
        pre_out = RepeatVector(config.max_answer_len)(pre_out)
        pre_out = GRU(config.hidden_dim, return_sequences=True)(pre_out)
        out = Dense(config.vocab_size, activation="softmax")(pre_out)

        self.model = Model([context_input, question_input], out)

    def compile(self):
        self.model.compile(Adam(), "categorical_crossentropy", metrics=["accuracy"])

    def load(self, file):
        self.model.load_weights(file)

    def generator_wrapper(self, gen):
        for c, q, a in gen:
            yield (
                [
                    c,
                    q
                ],
                np.stack([to_categorical(ans, self.config.vocab_size) for ans in a])
            )
