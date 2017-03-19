
# coding: utf-8

# In[ ]:

import itertools
import numpy as np
import scipy.ndimage

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, RMSprop
from keras import callbacks

np.random.seed(1337)  # for reproducibility

train_data_file = np.load("train.npy")

test_data_file = np.load("test.npy")

SIDE_LEN = 48

all_labels = set(train_data_file[:, 1])
classes_number = len(all_labels)
rename = dict(zip(
    sorted(set(all_labels)),
    range(classes_number)
))
rev_rename = {v: k for k, v in rename.items()}

prepared = train_data_file[train_data_file[:, 1].argsort()][(lambda label_count: list(itertools.chain.from_iterable(range(i, i+50) for i in [0]+label_count[:-1]))+list(itertools.chain.from_iterable(range(i+50, j) for i, j in zip([0]+label_count, label_count))))(list(itertools.accumulate(list(train_data_file[:, 1]).count(a) for a in rename.keys())))]

raw_train_data = prepared[25000:]
np.random.shuffle(raw_train_data)
raw_validation_data = prepared[:25000]
np.random.shuffle(raw_validation_data)


# Character generators:
def prepare(pic, side_len, mode='bilinear', threshold=0.4):
    z = np.ones((side_len, side_len))
    pic = scipy.misc.imresize(pic, side_len/max(pic.shape), interp=mode)/255
    pic = 1-(1-pic)/(1-pic).max()
    z[:pic.shape[0], :pic.shape[1]] = pic
    z = 1 - (1 - z < threshold)
    return z.reshape((side_len, side_len, 1))


def char_gen(array, size):
    i = 0
    if len(array.shape) == 2:
        source_pics = itertools.cycle(
            prepare(pic, SIDE_LEN) for pic in array[:, 0]
        )
        source_labels = itertools.cycle(
            to_categorical([rename[label]], classes_number)[0] for label in array[:, 1]
        )
    elif len(array.shape) == 1:
        source_pics = itertools.cycle(
            prepare(pic, SIDE_LEN) for pic in array
        )

    while True:
        if len(array.shape) == 2:
            yield (
                np.array(list(itertools.islice(source_pics, i, i + size))),
                np.array(list(itertools.islice(source_labels, i, i + size)))
            )
        elif len(array.shape) == 1:
            yield (
                np.array(list(itertools.islice(source_pics, i, i + size)))
            )

        i += size

conv_layers = [
    Convolution2D(32, 7, 7, activation='relu', border_mode='same', input_shape=(SIDE_LEN, SIDE_LEN, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2), border_mode='same'),

    Convolution2D(32, 7, 7, activation='relu', border_mode='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), border_mode='same'),

    Convolution2D(64, 5, 5, activation='relu', border_mode='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), border_mode='same'),

    Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2), border_mode='same'),

    Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
    BatchNormalization(),
]

full_layers = [
    Flatten(input_shape=(SIDE_LEN, SIDE_LEN, 1)),
    Dense(2**10),
    Activation('relu'),
    Dropout(0.3),
    Dense(classes_number),
    Activation('softmax'),
]

model = Sequential(conv_layers+full_layers)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adadelta(),
    metrics=['accuracy']
)

batch_size = 2000
epoch_number = 100

history = model.fit_generator(
    char_gen(train_data_file, batch_size),
    samples_per_epoch=10000,
    nb_epoch=epoch_number,
    #validation_data=char_gen(raw_validation_data, 250),
    #nb_val_samples=1250,
    verbose=1,
    #callbacks=[
    #    callbacks.EarlyStopping(monitor='val_loss', patience=7)
    #]
)

print(np.count_nonzero(np.array([rev_rename[guess] for guess in model.predict_classes(next(char_gen(raw_validation_data[:, 0], len(raw_validation_data))))]) == raw_validation_data[:, 1])/len(raw_validation_data))


def predict(model, test_data):
    prediction = model.predict_classes(test_data)
    prediction = '\n'.join(
        ','.join(map(str, line)) for line in enumerate([rev_rename[guess] for guess in prediction], 1)
    )
    with open('pred.csv', 'w') as pred:
        pred.write('Id,Category\n')
        pred.write(prediction)


predict(model, next(char_gen(test_data_file, len(test_data_file))))

print("Finished. Shutting down...")
