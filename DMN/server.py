from flask import Flask, render_template, request
from code.data import bAbIGen
from code.model import MyModel
import yaml
import numpy as np
import pandas as pd
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


def predict(text):
    pred = m.predict_from_text(text, vocab, rev_vocab)

    attns = (np.stack([p[0][-text.count("\n"):] for p in pred[1:]])[:, :, 0].T)

    frame = pd.DataFrame(
        [
            [
                '|style="background-color:#' +
                hex(num)[2:].upper().zfill(2)+"0000" +
                '"| !!' +
                str((num/255).round(2)) +
                "??"
                for num in line
            ] for line in (attns*255).astype('int')
        ], index=text.split("\n")[:-1]
    )

    return pred[0], frame.to_html().replace(">|", " ").replace("|", ">").replace("!!", '<font color="#FFFFFF">').replace("??", "</font>").replace("\\r", "")


@app.route('/', methods=['GET', 'POST'])
def print_form():
    if request.method == 'POST':
        if "ask" in request.form:
            answer, attentions = predict(request.form["fooput"].strip())
            return render_template(
                'form.html',
                answer=answer[0],
                text=request.form["fooput"].strip(),
                data=attentions
            )
        elif "load" in request.form:
            story = next(story_gen)
            answer, attentions = predict(story[0])
            return render_template(
                'form.html',
                text=story[0],
                answer=answer[0],
                true_answer=story[1],
                data=attentions
            )
    if request.method == 'GET':
        return render_template('form.html')


if __name__ == '__main__':
    with open("config.yml") as file:
        config = yaml.load(file)

    config = namedtuple("Config", config.keys())(*config.values())

    with open(config.vocab_file, "rb") as file:
        vocab, rev_vocab = pickle.load(file)

    m = MyModel(config, True)
    m.load(sys.argv[1])

    story_gen = TaskGen(config).generate()
    app.run()
