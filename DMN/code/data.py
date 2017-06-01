import regex as re
import numpy as np
import io

from keras.preprocessing.sequence import pad_sequences


def generate_from_string(source, config, vocab, concat=False):
    def tokenize(string):
        return [x.strip() for x in re.split('(\W+)?', string.lower()) if x.strip()]

    def vectorize_stories(stories, vocab):
        contexts, questions = zip(*stories)

        is_flattened = not isinstance(contexts[0][0], list)

        if not is_flattened:
            contexts = [
                [
                    [vocab[word] for word in sentence]
                    for sentence in context
                ]
                for context in contexts
            ]
            contexts = [
                pad_sequences(
                    sentences,
                    maxlen=config.max_sentence_len,
                    value=vocab["PAD"]
                )
                for sentences in contexts
            ]

            contexts = pad_sequences(
                contexts,
                maxlen=config.max_sentence_num,
                value=np.ones_like(contexts[0][0])*vocab["PAD"]
            )

        else:
            contexts = [
                [vocab[word] for word in context]
                for context in contexts
            ]

            contexts = pad_sequences(contexts, maxlen=config.max_context_len, value=vocab["PAD"])

        questions = [
            [vocab[word] for word in question]
            for question in questions
        ]
        questions = pad_sequences(questions, maxlen=config.max_question_len, value=vocab["PAD"])

        return list(zip(contexts, questions))

    file = io.StringIO(source)

    data = []
    story = []
    for line in file.readlines():
        line = line.strip()

        line = tokenize(line)
        if '?' in line:
            substory = None
            substory = [x for x in story if x]
            data.append((substory, line))

            stories = vectorize_stories(data, vocab)

            contexts, questions = zip(*stories)

            contexts = np.stack(contexts)
            questions = np.stack(questions)

            return [contexts, questions]
        else:
            if concat:
                story.extend(line)
            else:
                story.append(line)

    raise ValueError("No question")


class bAbIGen:
    def __init__(self, file_name, config):
        try:
            self.file = open(file_name)
            self.mode = "file"
        except:
            self.file = io.StringIO(file_name)
            self.mode = "string"
        self.config = config

    def __tokenize(self, string):
        return [x.strip() for x in re.split('(\W+)?', string.lower()) if x.strip()]

    def generate_stories(self, concat=False):
        data = []
        story = []
        for line in self.file.readlines():
            line = line.strip()

            if self.mode == "file":
                num, line = line.split(' ', 1)
                num = int(num)
                if num == 1:
                    story = []

            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.__tokenize(q)
                a = self.__tokenize(a + ".")
                substory = None
                substory = [x for x in story if x]
                data.append((substory, q, a))
            else:
                sent = self.__tokenize(line)
                if concat:
                    story.extend(sent)
                else:
                    story.append(sent)

        self.file.close()
        return data

    def generate_vocab(self):
        contexts, questions, answers = zip(*self.generate_stories(True))
        context_words = set().union(*(set(context) for context in contexts))
        question_words = set().union(*(set(question) for question in questions))
        answer_words = set().union(*(set(answer) for answer in answers))

        words = set().union(context_words, question_words, answer_words)

        rev_vocab = dict(enumerate(["PAD"] + list(words)))
        vocab = {word: number for number, word in rev_vocab.items()}

        return vocab, rev_vocab

    def vectorize_stories(self, stories, vocab):
        contexts, questions, answers = zip(*stories)

        is_flattened = not isinstance(contexts[0][0], list)

        if not is_flattened:
            contexts = [
                [
                    [vocab[word] for word in sentence]
                    for sentence in context
                ]
                for context in contexts
            ]
            contexts = [
                pad_sequences(
                    sentences,
                    maxlen=self.config.max_sentence_len,
                    value=vocab["PAD"]
                )
                for sentences in contexts
            ]

            contexts = pad_sequences(
                contexts,
                maxlen=self.config.max_sentence_num,
                value=np.ones_like(contexts[0][0])*vocab["PAD"]
            )

        else:
            contexts = [
                [vocab[word] for word in context]
                for context in contexts
            ]

            contexts = pad_sequences(contexts, maxlen=self.config.max_context_len, value=vocab["PAD"])
        questions = [
            [vocab[word] for word in question]
            for question in questions
        ]
        questions = pad_sequences(questions, maxlen=self.config.max_question_len, value=vocab["PAD"])
        answers = [
            [vocab[word] for word in answer]
            for answer in answers
        ]
        answers = pad_sequences(answers, maxlen=self.config.max_answer_len, value=vocab["PAD"])

        return list(zip(contexts, questions, answers))

    def generate_batches(self, batch_size, vocab, concat=False):
        stories = self.vectorize_stories(self.generate_stories(concat=concat), vocab)

        while True:
            np.random.shuffle(stories)
            for i in range(0, len(stories), batch_size):
                batch = stories[i:i+batch_size]

                contexts, questions, answers = zip(*batch)

                contexts = np.stack(contexts)
                questions = np.stack(questions)
                answers = np.stack(answers)

                yield contexts, questions, answers
