import sys
import numpy as np
import os

np.random.seed(1337)

pairs = []
with open(sys.argv[1]) as train_data_file:
    for line in train_data_file:
        base, *variants = line.split()
        for var in variants:
            pairs.append((" ".join(base), var.replace('_', ' ')))
np.random.shuffle(pairs)

words, pronunciations = zip(*pairs)

words_vocab = list(enumerate(sorted(set.union(*(set(word.split()) for word in words)))))
prons_vocab = list(enumerate(sorted(set.union(*(set(word.split()) for word in pronunciations)))))

SPLIT = 0.85
train_words, validation_words = np.split(
    words,
    [int(len(words)*SPLIT)]
)
train_pronunciations, validation_pronunciations = np.split(
    pronunciations,
    [int(len(words)*SPLIT)]
)

with open(os.path.join('prepared', 'words_vocab'), 'w') as file:
    file.write("\n".join("\t".join(list(map(str, pair))[::-1]) for pair in words_vocab))
with open(os.path.join('prepared', 'prons_vocab'), 'w') as file:
    file.write("\n".join("\t".join(list(map(str, pair))[::-1]) for pair in prons_vocab))
with open(os.path.join('prepared', 'train_words'), 'w') as file:
    file.write("\n".join(train_words))
with open(os.path.join('prepared', 'train_pronunciations'), 'w') as file:
    file.write("\n".join(train_pronunciations))
with open(os.path.join('prepared', 'validation_words'), 'w') as file:
    file.write("\n".join(validation_words))
with open(os.path.join('prepared', 'validation_pronunciations'), 'w') as file:
    file.write("\n".join(validation_pronunciations))

test = []
with open(sys.argv[2]) as test_data_file:
    test_data_file.readline()
    for line in test_data_file:
        _, base = line.strip().split(',')
        test.append(" ".join(base))

with open(os.path.join('prepared', 'test_words'), 'w') as file:
    file.write("\n".join(test))

print(len(train_words))
print(len(test))
