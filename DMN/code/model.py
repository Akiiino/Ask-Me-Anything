from keras import backend as K
from keras.layers import (
    Input, Dense, GRU, RepeatVector, concatenate, Lambda, Layer, TimeDistributed,
    BatchNormalization, multiply, Bidirectional, Activation
)

import tensorflow as tf
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Model
from keras.optimizers import Adadelta


class MyEmbedding(Layer):
    def __init__(self, vocab_size, embedding_dim, regularizer=None, **kwargs):
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.regularizer = regularizer
        super(MyEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.embedding = self.add_weight(
            shape=(self.vocab_size, self.embedding_dim),
            initializer="uniform",
            trainable=True,
            regularizer=self.regularizer,
            name="weight"
        )
        super(MyEmbedding, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        return tuple(shape + [self.embedding_dim])

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embedding, inputs)


class Positioning(Layer):
    def __init__(self):
        self.cache = None
        super(Positioning, self).__init__()

    def call(self, inputs):
        if self.cache is None:
            self.cache = K.ones_like(inputs, dtype="int32") * K.arange(inputs.shape[-1])

        return self.cache


class MyModel:
    def __init__(self, config):
        self.config = config

        question_input = Input((config.max_question_len, ), dtype='int32')

        embedded_question = MyEmbedding(config.vocab_size, config.hidden_dim)(question_input)
        summarized_question = Bidirectional(
            GRU(
                config.hidden_dim,
                activation='linear',
                implementation=2,
                recurrent_dropout=0.1
            )
        )(embedded_question)

        summarized_question = BatchNormalization()(summarized_question)
        summarized_question = Activation('tanh')(summarized_question)
        repeated_question = RepeatVector(config.max_sentence_num)(summarized_question)

        context_input = Input((config.max_sentence_num, config.max_sentence_len), dtype='int32')
        embedded_context = MyEmbedding(config.vocab_size, config.hidden_dim)(context_input)
        positions = Positioning()(context_input)
        embedded_positions = MyEmbedding(config.max_sentence_len, config.hidden_dim)(positions)

        embedded_context = multiply([embedded_context, embedded_positions])

        embedded_context = Lambda(lambda x: K.sum(x, axis=2))(embedded_context)
        contexts = Bidirectional(
            GRU(
                config.hidden_dim,
                return_sequences=True,
                activation='linear',
                implementation=2,
                recurrent_dropout=0.1
            )
        )(embedded_context)
        contexts = BatchNormalization()(contexts)
        contexts = Activation('tanh')(contexts)

        memory = summarized_question

        for pass_ in range(4):
            repeated_memory = RepeatVector(config.max_sentence_num)(memory)
            attention = concatenate([
                contexts,
                repeated_question,
                repeated_memory,
                multiply([
                    contexts,
                    repeated_memory
                ]),
                multiply([
                    contexts,
                    repeated_question
                ]),
                Lambda(lambda x: K.abs(x[0]-x[1]))([
                    contexts,
                    repeated_memory
                ]),
                Lambda(lambda x: K.abs(x[0]-x[1]))([
                    contexts,
                    repeated_question
                ])
            ])

            attention = TimeDistributed(Dense(512, activation='relu'))(attention)
            attention = TimeDistributed(Dense(1, activation='sigmoid'))(attention)
            attention = Lambda(lambda x: K.concatenate([x]*config.hidden_dim*2))(attention)

            attented_contexts = multiply([attention, contexts])
            episode = Lambda(lambda x: K.sum(x, axis=1))(attented_contexts)
            episode = concatenate([episode, memory])

            memory = Dense(config.hidden_dim*2, activation='tanh')(episode)

        pre_out = Dense(config.hidden_dim, activation="linear")(memory)
        pre_out = BatchNormalization()(pre_out)
        pre_out = Activation('relu')(pre_out)
        pre_out = RepeatVector(config.max_answer_len)(pre_out)
        pre_out = GRU(config.hidden_dim, return_sequences=True, implementation=2)(pre_out)
        out = Dense(config.vocab_size, activation="softmax")(pre_out)

        self.model = Model([context_input, question_input], out)

    def compile(self):
        self.model.compile(Adadelta(), "categorical_crossentropy", metrics=["accuracy", self.accuracy])

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

    def accuracy(self, true, pred):
        return K.mean(K.equal(K.mean(K.equal(K.argmax(true, -1), K.argmax(pred, -1)), -1), 1))
