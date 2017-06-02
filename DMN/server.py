from flask import Flask, render_template, request
from code.data import bAbIGen
from code.model import MyModel
import yaml
import numpy as np
import pickle
import sys
from collections import namedtuple

app = Flask(__name__)


class TaskGen():
    def __init__(self, config):
        self.stories = bAbIGen("tasks/test_all.txt", config).generate_stories(lowercase=False)

    def __textify(self, l):
        return ((
            "\n".join(" ".join(line) for line in l[0]) + "\n" + " ".join(l[1])
        ).replace(" .", ".").replace(" ,", ",").replace(" ?", "?"),
            " ".join(l[2]).replace(" .", ".").replace(" ,", ",")
        )

    def generate(self):
        while True:
            story = np.random.choice(len(self.stories))

            yield self.__textify(self.stories[story])


@app.route('/', methods=['GET', 'POST'])
def print_form():
    if request.method == 'POST':
        if "ask" in request.form:
            return render_template(
                'form.html',
                answer=m.predict_from_text(request.form["fooput"], vocab, rev_vocab),
                text=request.form["fooput"]
            )
        elif "load" in request.form:
            story = next(story_gen)
            return render_template(
                'form.html',
                text=story[0],
                answer=m.predict_from_text(story[0], vocab, rev_vocab),
                true_answer=story[1]
            )
    if request.method == 'GET':
        return render_template('form.html')


if __name__ == '__main__':
    with open("config.yml") as file:
        config = yaml.load(file)

    config = namedtuple("Config", config.keys())(*config.values())

    with open(config.vocab_file, "rb") as file:
        vocab, rev_vocab = pickle.load(file)

    m = MyModel(config)
    m.load(sys.argv[1])

    story_gen = TaskGen(config).generate()
    app.run()
