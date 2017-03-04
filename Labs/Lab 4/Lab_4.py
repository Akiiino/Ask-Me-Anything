
# coding: utf-8

# # Lab 4. Word embeddings.

# Для начала, как всегда, всевозможные приготовления к работе. Использовать будем заранее предобработанные текты из трёх википедий — английской, русской и simple. Большая часть лабы выполнялась на full, но переключение на любую другую производится заменой единственного параметра (не то чтобы это было неочевидно, впрочем).

# In[1]:

import collections
import math
import os
import random
from datetime import datetime
from itertools import accumulate
import itertools
from IPython.display import clear_output
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf
import pickle
import stop_words
import regex as re
import sys

import scipy
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import sklearn.decomposition
from sklearn.manifold import TSNE


# In[2]:

class Wiki:
    def __init__(self, wiki):
        self.wiki = wiki
        self.__set_consts()
        
        if not os.path.exists(wiki):
            os.mkdir(wiki)
        
        self.counter = self.vocab = self.rev_vocab = None
        self.counter, self.vocab, self.rev_vocab = self.__get_vocab()
        

    def __set_consts(self):
        wikis = {
            'full': {
                'lang'      : 'en',
                'wiki_file' : '/Users/akiiino/Documents/Wiki/full/processed.dat',
                'vocab_size': 150000
            },
            'simple': {
                'lang'      : 'en',
                'wiki_file' : '/Users/akiiino/Documents/Wiki/simple/processed.dat',
                'vocab_size': 50000
            },
            'ru': {
                'lang'      : 'ru',
                'wiki_file' : '/Users/akiiino/Documents/Wiki/ru/processed.dat',
                'vocab_size': 100000
            },
        }

        
        self.LANG = wikis[self.wiki]['lang']
        self.WIKI_FILE = wikis[self.wiki]['wiki_file']
        self.VOCAB_SIZE = wikis[self.wiki]['vocab_size']
        
        self.WORD_COUNT_FILE = os.path.join(self.wiki, 'word_counts')
        self.LSA_TFIDF_FILE = os.path.join(self.wiki, 'LSA_TfIdf.npz')
        self.LSA_EMBED_FILE = os.path.join(self.wiki, 'LSA_embed.npy')
        self.LSA_VOCAB_FILE = os.path.join(self.wiki, 'LSA_vocab')
        self.W2V_EMBED_FILE = os.path.join(self.wiki, 'W2V_embed.npy')
        self.GLOVE_COOCC_FILE = os.path.join(self.wiki, 'glove_coocc.npz')
        self.GLOVE_EMBED_FILE = os.path.join(self.wiki, 'glove_embed.npy')
        self.TSNE_FILE = os.path.join(self.wiki, 'tsne.npy')
        
        self.stop_words = set(stop_words.get_stop_words(self.LANG))
        self.punct_replace = re.compile(r"\p{P}+")


    def _normalize(self, text):
        text = text.lower()
        text = self.punct_replace.sub("", text)
        text = (word for word in text.split() if word not in self.stop_words)
        return text


    def wiki_gen(self, use_vocab=True, chunk_size=2**26, single_pass=False, joined=False):
        read = 0

        with open(self.WIKI_FILE) as wiki_file:
            while True:
                lines = " ".join(wiki_file.readlines(chunk_size))
                read += len(lines)

                if chunk_size >= 2**20:
                    print('Pulling... total {} MB\r'.format(read//2**20), end="")

                if not lines:
                    if single_pass:
                        return
                    wiki_file.seek(0)
                    print("Starting over...          ")
                    continue

                lines = self._normalize(lines)

                if self.vocab is not None and use_vocab:
                    lines = (self.vocab[word] if word in self.vocab else 0 for word in lines)

                if not joined:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
                    yield lines
                else:
                    yield " ".join(lines)
        
    def __get_vocab(self):
        if not os.path.exists(self.WORD_COUNT_FILE):
            counter = collections.Counter()

            for chunk in self.wiki_gen(use_vocab=False, single_pass=True):
                counter.update(chunk)

            with open(self.WORD_COUNT_FILE, 'wb') as file:
                pickle.dump(counter, file)

        else:
            with open(self.WORD_COUNT_FILE, 'rb') as file:
                counter = collections.Counter(pickle.load(file))


        unk_count = sum(counter.values()) - sum(x[1] for x in counter.most_common(self.VOCAB_SIZE - 1))
        counter = dict([('UNK', unk_count)] + counter.most_common(self.VOCAB_SIZE - 1))
        
        vocab     = {word: num for num, (word, _) in enumerate(counter.items())}
        rev_vocab = list(vocab.keys())
        
        return counter, vocab, rev_vocab


# In[3]:

wiki = Wiki("simple")


# ## LSA

# ### **Задание**
# **1**.  Вычислите $X$ понравившимся способом (возможно, стоит использовать разреженную матрицу).
# 
# **2**. Обучите представления слов при $k = 128$ на своем корпусе.

# Обучение устроено просто: пайплайн из `sklearn`'ового `TfidfVectorizer` для построения самой матрицы $X$, после чего Truncated SVD в виде `scipy.sparse.linalg.svds`.

# In[4]:

def build_LSA(wiki):
    vectorizer = TfidfVectorizer(vocabulary=wiki.vocab)
    tf_idf_matrix = vectorizer.fit_transform(
        wiki.wiki_gen(
            chunk_size=1,
            use_vocab=False,
            single_pass=True,
            joined=True
        )
    ).T
    np.savez(wiki.LSA_TFIDF_FILE, tf_idf_matrix)

    print("Tf-Idf done")
    
    svd = sklearn.decomposition.TruncatedSVD(n_components=128)
    
    LSA_embeds = svd.fit_transform(tf_idf_matrix)
    print("SVD done")
    
    return LSA_embeds


# In[5]:

def get_LSA(wiki):
    if not os.path.exists(wiki.LSA_EMBED_FILE):
        LSA_embeds = build_LSA(wiki)
        np.save(wiki.LSA_EMBED_FILE, LSA_embeds)
    else:
        LSA_embeds = np.load(wiki.LSA_EMBED_FILE)
        
    return LSA_embeds


# In[6]:

LSA_embeds = get_LSA(wiki)


# При этом после вычисления всё сохраняется на диск, чтобы лишний раз не считать, если понадобится снова.

# **3**. Реализуйте поиск k-ближайших соседей для Евклидовой меры в 128-мерном пространстве .

# Скелет для нахождения соседей: по вектору находим ближайшие слова; сложность линейная, т.к. использован `argpartition`. Сам вектор может быть получен как напрямую из слова, так и, например, сложениями и вычитаниями других векторов.

# In[7]:

def closest_to_vec(vec, embeds, wiki, metric_name='euclid', count=10, with_dists=False):
        metrics = {
            'euclid': euclidean_distances,
            'cosine': cosine_similarity,
            'manhattan': manhattan_distances
        }
        
        metric = metrics[metric_name]
        
        dists = metric(
                [vec],
                embeds
        )[0]
        
        if metric_name == "cosine":
                closest_indexes = dists.argpartition(range(-count, 0))[-1:-count-1:-1]
        else:
                closest_indexes = dists.argpartition(range(0, count))[:count]

        if with_dists:
            return list(zip(dists[closest_indexes], [wiki.rev_vocab[x] for x in closest_indexes if x]))
        else:
            return [wiki.rev_vocab[x] for x in closest_indexes if x]


# Сам поиск по словам:

# In[8]:

def closest_words(word, embeds, wiki, metric='euclid', count=10, with_dists=False):
    return closest_to_vec(embeds[wiki.vocab[word]], embeds, wiki, metric, count, with_dists)


# In[9]:

words = ["field", "directed", "financial", "road", "provides", "player", "2011", "edition", "battle", "ended", "son", "least", "mexico", "male", "medal", "big", "central", "according", "km", "year", "rights", "george", "founded", "tournament", "instead", "movie", "445", "system", "york", "win"]


# **4**. Найдите 10 ближайших представлений к 30 словам, которые выбрали сами. Попытайтесь сделать так, чтобы примеры были интересными.

# In[10]:

for word in words:
    print(word.ljust(10), ":", " ".join(closest_words(word, LSA_embeds, wiki)[1:]))


# Есть несколько интересных вещей. 
# Например, следующее:
# - То, как сгруппированы числа (а особенно то, что рядом находятся числа схожей величины);
# - то, что рядом со словом "york" находятся также и "jersey", и "new", с которого обычно начинаются оба;
# - то, как "central" явно встречается преимущественно в географическом контексте;
# - Википедия явно очень беспокоится о правах в Латвии;
# - среди мужских имён почему-то затесалось "wife".

# **5**. Проделайте то же самое для косинусной меры. Какой результат вам показался более интересным?
# 

# In[11]:

for word in words:
    print(word.ljust(10), ":", " ".join(closest_words(word, LSA_embeds, wiki, metric="cosine")[1:]))


# Тут вообще непонятно что происходит. Видимо, в длине векторов тоже есть важная информация и лучше было всё же с евклидовой метрикой.

# **6**. Предложите свою меру длины, проверьте, как она работает.

# In[12]:

for word in words:
    print(word.ljust(10), ":", " ".join(closest_words(word, LSA_embeds, wiki, metric="manhattan")[1:]))


# В целом, примерно как и в евклидовой метрике.

# ## Word2Vec

# Сначала о теории:
# 
# ### **Задание**
# 
# **1**. Как можно представить модель skip-gram в виде нейронной сети?

# Тут практически нет отличий от CBOW. Два линейных слоя без нелинейности между ними, в конце --- softmax. На вход сеть принимает one-hot вектор размерности $W$, кодирующий центральное слово; на выходе --- вектор той же размерности, представляющий собой вероятности каждого слова.

# **2**. Оцените сложность обучения skip-gram модели относительно параметров 
# * $T$ - размер корпуса;
# * $W$ - размер словаря;
# * $c$ - радиус окна;
# * $d$ - размерность представлений;

# ** Forward propagation:**
# 
# $(d \times W) \cdot (W \times 1); (W \times d) \cdot (d \times 1) \implies O(WD) + O(WD) = O(WD)$;
# 
# Softmax --- ещё дополнительно линейное (от $W$) время. Итого $O(WD)$ --- ничего страшного.
# 
# ** Backpropagation:**
#     
# Взятие производных функции ошибки --- $O(Wdc)$, т.к. мы берём производные по матрицам размера $O(Wd)$, и наш размер контекста --- $c$. При этом нам надо это будет сделать $O(T)$ раз --- для каждого батча.
# 
# Итого --- $O(TWdc)$

# Батчи генерируются из скользящего окна шириной `context_window`; из каждого окна берётся `context_examples` пар слов.

# In[13]:

def w2v_batch_gen(batch_size, context_examples, context_window):
    assert batch_size % context_examples == 0
    assert context_examples <= 2 * context_window

    gen = wiki.wiki_gen()
    
    pulled = 0
    buffer = collections.deque()

    while True:
        while len(buffer) < 2 * context_window + batch_size//context_examples:
            buffer.extend(next(gen))
        
        batch  = np.ndarray(shape=(batch_size),    dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        span = 2 * context_window + 1

        for i in range(batch_size // context_examples):
            target = context_window
            available_targets = list(range(span))
            available_targets.remove(target)
            for j in range(context_examples):
                target = random.choice(available_targets)
                available_targets.remove(target)
                batch[i * context_examples + j] = buffer[context_window]
                labels[i * context_examples + j, 0] = buffer[target]
            buffer.popleft()
        yield batch, labels


# ### **Задание**
# 1. Выберите понравивщуюся вам функцию потерь. Оцените сложность обучения для нее, сравните с простым softmax.
# 2. На основе оценки сложности определиться с количеством эпох, размера вектора $d$ = 256 будет достаточно. Определитесь с размером окна. Будьте готовы, что на большом корпусе обучение может длиться около суток.

# Пользоваться будем NCE --- преимущественно из-за простоты реализации. Сложность обучения оценить легко: фактически, единственное существенное изменение (асимптотики, разумеется; сами алгоритмы отличаются существенно) заключается в том, что вместо прохода по всем $W$ словам, мы рассматриваем только $k$ cлучайных слов; итого сложность $O(Tcdk)$.

# In[14]:

batch_size = 256
embedding_dim = 256
context_window = 6
context_examples = 4
neg_samples = 100

w2v_graph = tf.Graph()
with w2v_graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    all_embeddings = tf.Variable(
        tf.random_uniform(
            [wiki.VOCAB_SIZE, embedding_dim],
            -1.0,
            1.0
        )
    )
    embed = tf.nn.embedding_lookup(all_embeddings, train_inputs)

    weights = tf.Variable(
        tf.truncated_normal(
            [wiki.VOCAB_SIZE, embedding_dim],
            stddev=1.0 / math.sqrt(embedding_dim)
        )
    )
    biases = tf.Variable(tf.zeros([wiki.VOCAB_SIZE]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=weights,
            biases=biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=neg_samples,
            num_classes=wiki.VOCAB_SIZE
        )
    )

    optimizer = tf.train.AdagradOptimizer(0.5).minimize(loss)

    init = tf.global_variables_initializer()


# Экспериментальным путём выяснено, что десяти миллионов итераций достаточно, чтобы обучиться (по крайней мере до состояния, где заметных улучшений нет). Занимает это (у меня, по крайней мере) около восьми-девяти часов (надо бы, конечно, сделать две-три полные эпохи, но это очень уж долго. Может быть, потом). Девять часов --- это всё-таки немало, так что тут тоже есть возможность загружать и выгружать результаты.

# **3**. Обучите skip-gram модель.

# In[15]:

def build_w2v(wiki):
    num_steps = 500

    with tf.Session(graph=w2v_graph) as session:
        init.run()

        average_loss = 0
        batch_gen = w2v_batch_gen(batch_size, context_examples, context_window)
        for step in range(num_steps):
            try:
                for _ in range(10000):
                    batch_inputs, batch_labels = next(batch_gen)
                    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                    _, loss_val= session.run([optimizer, loss], feed_dict=feed_dict)

                    average_loss += loss_val


                clear_output()
                print("Time: {}; steps: {}; avg. loss: {}.".format(datetime.now().strftime("%H:%M:%S"), step*10000, average_loss/10000))
                average_loss = 0
                w2v_embeds = all_embeddings.eval()
                for i in range(10):
                    word = wiki.rev_vocab[random.randint(1, 150)]
                    print(word.ljust(10), ":", " ".join(closest_words(word, w2v_embeds, wiki)[1:]))

            except KeyboardInterrupt:
                break
        w2v_embeds = all_embeddings.eval()

    return w2v_embeds


# In[ ]:

def get_w2v(wiki):
    if not os.path.exists(wiki.W2V_EMBED_FILE):
        w2v_embeds = build_w2v(wiki)
        np.save(wiki.W2V_EMBED_FILE, w2v_embeds)
    else:
        w2v_embeds = np.load(wiki.W2V_EMBED_FILE)
        
    return w2v_embeds


# In[ ]:

w2v_embeds = get_w2v(wiki)


# **4**. Попробуйте снова найти ближайшие представления для тех 30 слов. Улучшился ли результат визуально? Попробуйте разные меры расстояния (евклидова, косинусная).

# In[ ]:

for word in words:
    print(word.ljust(10), ":", " ".join(closest_words(word, w2v_embeds, wiki)[1:]))


# Видно, что качество сильно улучшилось; вместо слов, которые относятся примерно к одной теме теперь рядом находятся слова, значащие примерно одно и то же; вместо всяких "founded" --- "university" и прочее, теперь тут "founded: --- "formed", "established"... "bought"... В общем, теперь смысла гораздо больше (что было ясно ещё из сложений-вычитаний).

# **5**. Найдите ближайшие вектора для выражения v(king) - v(man) + v(women). Если модель обучена хорошо, то среди ближайших векторов должно быть представление v(queen). 
# 

# Функция для сложения-вычитания векторов:

# In[ ]:

def analogy(pos, neg, embeds, wiki, metric='euclid', count=10):
    pos_vec = [embeds[wiki.vocab.get(w, 0)] for w in pos]
    neg_vec = [embeds[wiki.vocab.get(w, 0)] for w in neg]

    return [
        pair for pair in closest_to_vec(
            sum(pos_vec) - sum(neg_vec),
            embeds,
            wiki,
            metric_name=metric,
            count=count,
            with_dists=True
        ) if pair[1] not in pos and pair[1] not in neg
    ]


# Можно поэкспериментировать и посмотреть, что получится. Базовые примеры:

# In[ ]:

analogy(["king", "woman"], ["man"], w2v_embeds, wiki)


# **6**. На основе арифметических операций над представлениями предложите алгоритмы, которые
#   * Для страны определяют столицу
#   * Для компании определяют CEO
#   * Для прилагательного определяют превосходную форму
#   * Для слова определяют множественное число
#   * Для слова находит его антоним

# #1. Столицы:

# In[ ]:

def get_capitals(countries):
    return [
        analogy(
            [country,
            "moscow", "seoul", "athens",],
            ["russia", "korea", "greece",],
            w2v_embeds,
            wiki
        )[0][1] for country in countries
    ]

countries = [
    "germany",
    "france",
    "japan",
    "spain",
    "canada",
    "greece",
    "albania",
    "afghanistan",
    "algeria",
    "norway",
    "india"
]
print("\n".join("\t".join(pair) for pair in zip(countries, get_capitals(countries))))


# Как видно, работает, но не всегда идеально (Афганистан, Чикаго, што?). Чем больше пар "страна --- столица" взято в качестве опоры --- тем лучше качество (но это только гипотеза, впрочем), но тем больше вопросов "А зачем вообще, если есть большая база столиц?".
# 
# Логично предположить, что на самом деле вектор "Москва" - "Россия" + "Сеул" - "Корея" + "Афины" - "Греция" в три раза больше "эталонного" сдвига между столицей и страной, однако если этот вектор поделить на три, всё только портится. Видать, не такое уж и векторное пространство.

# #2. CEO:

# In[ ]:

def get_CEOs(companies):
    return [
        analogy(
            ["zuckerberg", company],
            ["facebook"],
            w2v_embeds,
            wiki
        )[0][1] for company in companies
    ]

companies = [
    "apple",
    "sun",
    "oracle"
]
print("\n".join(" ".join(pair) for pair in zip(companies, get_CEOs(companies))))


# Мда. Идея должна бы срaботать, но нужно нечеловеческое время на обучение и словарь побольше.

# #3. Прилагательные:

# In[ ]:

def get_superlatives(adjectives):
    return [
        analogy(
            ["fastest", adjective],
            ["fast"],
            w2v_embeds,
            wiki
        )[0][1] for adjective in adjectives
    ]

adjectives = [
    "dark",
    "big",
    "good",
    "sharp",
    "strange",
    "blue",
]
print("\n".join(" ".join(pair) for pair in zip(adjectives, get_superlatives(adjectives))))


# О, работает! Почти... Смысл точно местами прослеживается.

# #4. Множественное число:

# In[ ]:

def get_plurals(nouns):
    return [
        analogy(
            ["boxes", noun],
            ["box"],
            w2v_embeds,
            wiki
        )[0][1] for noun in nouns
    ]

nouns = [
    "queen",
    "moose",
    "box",
    "chair",
    "glass",
]
print("\n".join(" ".join(pair) for pair in zip(nouns, get_plurals(nouns))))


# Не очень что-то.

# ## GloVe

# Построим матрицу:

# In[ ]:

def build_cooccur(wiki, context_window=5):
    chunk_size = 50000

    slices = list(range(a, b) for a, b in (lambda x: zip(x, x[1:]))(np.arange(0, wiki.VOCAB_SIZE+chunk_size, chunk_size)))

    mat = scipy.sparse.lil_matrix((wiki.VOCAB_SIZE, wiki.VOCAB_SIZE), dtype=np.float32)

    for x, range_x in enumerate(slices):
        for y, range_y in list(enumerate(slices))[x:]:
            local_mat = np.zeros((chunk_size, chunk_size), dtype=np.float32)
            print("starting chunk {}, {}".format(x, y))

            text_gen = wiki.wiki_gen(single_pass=True)

            buffer = collections.deque()
            pulled = 0

            stop = False
            i = 0
            while True:
                i+= 1;
                if i % 100000 == 0:
                    print("{}\r".format(i), end="")
                while len(buffer) < context_window + 1:
                    try:
                        next_chunk = next(text_gen)
                    except StopIteration:
                        stop = True
                        break
                    buffer.extend(next_chunk)

                if stop:
                    break

                for j in range(1, context_window+1):
                    row, column = sorted([buffer[0], buffer[j]])
                    if row in range_x and column in range_y:
                        if i == j:
                            local_mat[row - x*chunk_size, column - y*chunk_size] += 1./j
                            local_mat[column - y*chunk_size, row - x*chunk_size] += 1./j
                        else:
                            local_mat[row - x*chunk_size, column - y*chunk_size] += 1./j
                buffer.popleft()

            mat[x*chunk_size:(x+1)*chunk_size, y*chunk_size:(y+1)*chunk_size] = local_mat
            if x!= y:
                mat[y*chunk_size:(y+1)*chunk_size, x*chunk_size:(x+1)*chunk_size] = local_mat.T
    mat = scipy.sparse.lil_matrix(mat)
    return mat


# In[ ]:

def get_cooccur(wiki):
    if not os.path.exists(wiki.GLOVE_COOCC_FILE):
        cooccurrences_matrix = build_cooccur(wiki)
        np.savez(wiki.GLOVE_COOCC_FILE, cooccurrences_matrix)
        
    else:
        cooccurrences_matrix = np.load(wiki.GLOVE_COOCC_FILE).items()[0][1].item()

    return cooccurrences_matrix

cooccurrences_matrix = get_cooccur(wiki)


# In[ ]:

def glove_batch_gen(cooccur_mat, batch_size):
    def single_gen():
        for i, (row, data) in enumerate(zip(cooccur_mat.rows, cooccur_mat.data)):
            for data_idx, j in enumerate(row):
                yield i, j, data[data_idx]
    
    batches = list(single_gen())
                
    while True:
        np.random.shuffle(batches)
        for batch in range(0, len(batches), batch_size):
            yield zip(*batches[batch:batch+batch_size])


# In[ ]:

glove_graph = tf.Graph()

with glove_graph.as_default():
    count_max      = tf.constant([cooccurrence_cap], dtype=tf.float32, name='max_cooccurrence_count')
    scaling_factor = tf.constant([scaling_factor_c], dtype=tf.float32, name="scaling_factor")

    focal_input        = tf.placeholder(tf.int32,   shape=[batch_size], name="focal_words")
    context_input      = tf.placeholder(tf.int32,   shape=[batch_size], name="context_words")
    cooccurrence_count = tf.placeholder(tf.float32, shape=[batch_size], name="cooccurrence_count")

    focal_embeddings = tf.Variable(
        tf.random_uniform(
            [wiki.VOCAB_SIZE, embedding_dim],
            1.0,
            -1.0
        ),
        name="focal_embeddings"
    )
    context_embeddings = tf.Variable(
        tf.random_uniform(
            [wiki.VOCAB_SIZE, embedding_dim],
            1.0,
            -1.0
        ),
        name="context_embeddings"
    )

    focal_biases   = tf.Variable(tf.random_uniform([wiki.VOCAB_SIZE], 1.0, -1.0), name='focal_biases')
    context_biases = tf.Variable(tf.random_uniform([wiki.VOCAB_SIZE], 1.0, -1.0), name="context_biases")

    focal_embedding   = tf.nn.embedding_lookup([focal_embeddings],   focal_input)
    context_embedding = tf.nn.embedding_lookup([context_embeddings], context_input)
    focal_bias        = tf.nn.embedding_lookup([focal_biases],       focal_input)
    context_bias      = tf.nn.embedding_lookup([context_biases],     context_input)

    weighting_factor = tf.minimum(
        1.0,
        tf.pow(
            tf.div(
                cooccurrence_count,
                count_max),
            scaling_factor
        )
    )

    embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

    log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))

    distance_expr = tf.square(tf.add_n([
        embedding_product,
        focal_bias,
        context_bias,
        tf.negative(log_cooccurrences)]
    ))

    glove_losses = tf.multiply(weighting_factor, distance_expr)
    glove_loss = tf.reduce_sum(glove_losses)
    optimizer = tf.train.AdagradOptimizer(0.15).minimize(glove_loss)

    combined_embeddings = tf.add(focal_embeddings, context_embeddings, name="combined_embeddings")

    init = tf.global_variables_initializer()


# In[ ]:

cooccurrence_cap = 250
scaling_factor_c = 0.75
batch_size = 512
embedding_dim = 256


# In[ ]:

def build_glove(wiki):
    num_steps = 750
    average_loss = 0

    with tf.Session(graph=glove_graph) as session:
        init.run()
        for step in range(num_steps):
            try:
                batch_gen = glove_batch_gen(cooccurrences_matrix, batch_size)
                for _ in range(1000):
                    centers, contexts, counts = next(batch_gen)
                    if len(counts) != batch_size:
                        continue
                    feed_dict = {
                        focal_input: centers,
                        context_input: contexts,
                        cooccurrence_count: counts
                    }
                    _, loss_val = session.run([optimizer, glove_loss], feed_dict=feed_dict)
                    average_loss += loss_val
                clear_output()
                print(
                    "Time: {}; steps: {}; avg. loss: {}.".format(
                        datetime.now().strftime("%H:%M:%S"),
                        step*1000,
                        average_loss/1000)
                )
                average_loss = 0
                glove_embeds = combined_embeddings.eval()
                for i in range(10):
                    w = wiki.rev_vocab[random.randint(1, 150)]
                    print(w.ljust(10), ":", " ".join(closest_words(w, glove_embeds, wiki)[1:]))
            except KeyboardInterrupt:
                break
        glove_embeds = combined_embeddings.eval()
    return glove_embeds


# In[ ]:

def get_glove(wiki):
    if not os.path.exists(wiki.GLOVE_EMBED_FILE):
        glove_embeds = build_glove(wiki)
        np.save(wiki.GLOVE_EMBED_FILE, glove_embeds)
    else:
        glove_embeds = np.load(wiki.GLOVE_EMBED_FILE)
    return glove_embeds


# In[ ]:

glove_embeds = get_glove(wiki)


# Теперь T-SNE. Тут тоже просто библиотечная реализация. Единственное, если вогнать всю матрицу, падает от нехватки памяти, так что обойдёмся первыми 10к векторов.

# In[ ]:

tsne = TSNE(3)
post_tsne_glove = tsne.fit(glove_embeds[:10000])


# In[ ]:

get_ipython().magic('matplotlib notebook')
fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111, projection="3d")
for i, (label, _) in enumerate(collections.Counter(wiki.counter).most_common(100)):
    x, y, z = post_tsne_glove.embedding_[i]
    ax.scatter(x, y, z, s=9, alpha=0.1)
    ax.text(x, y, z, label)


# Не вполне ясно, что предлагается из этого выжимать, впрочем. Давайте просто померяем углы:

# In[ ]:

def angle(wiki, s1, w1, s2, w2):
    return np.arccos(cosine_similarity(
        [sum(post_tsne_glove.embedding_[wiki.vocab[x]]*sign for sign, x in zip(s1, w1))],
        [sum(post_tsne_glove.embedding_[wiki.vocab[x]]*sign for sign, x in zip(s2, w2))],
    ))[0][0]*90/np.pi


# In[ ]:

angle(wiki, [1, -1], ["man", "woman"], [1, -1], ["mr", "ms"])


# Более-менее

# In[ ]:

angle([1, -1], ["good", "best"], [1, -1], ["bad", "worst"])


# Тоже неплохо. Без CEO, правда, придётся обойтись --- в словаре их нет почти.

# Метрика для сравнения моделей --- количество правильных ответов на вопросы из датасета Миколова:

# In[ ]:

with open("questions-words.txt") as question_file:
    parsed = (
        line for line in (
            tuple(wiki._normalize(line)) for line in question_file.readlines() if not line.startswith(":")
        ) if len(line) == 4
    )

    *questions, labels = zip(*parsed)

    questions = list(zip(*questions))


# In[ ]:

def evaluate(embeds, wiki, questions, labels):
    total = 0
    for i, (label, (pos1, neg, pos2)) in enumerate(zip(labels, questions)):
        if i % 50 == 0:
            print("{}\r".format((i, total)), end="")
        total += (label in set(x[1] for x in analogy([pos1, pos2], [neg], embeds, wiki)))
    return total


# In[ ]:

evaluate(LSA_embeds, wiki, questions, labels)


# In[ ]:

evaluate(w2v_embeds, wiki, questions, labels)


# In[ ]:

evaluate(glove_embeds, glove_vocab, questions, labels)


# Ну. Могло быть лучше, конечно, но w2v всех обошёл в ~2 раза
