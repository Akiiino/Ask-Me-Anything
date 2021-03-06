# Ask Me Anything
### Попов Никита Сергеевич, группа 151-2

## 0. Disclaimer

  В марте 2016 года на arXiv.org была опубликована статья [Ask Me Anything: Dynamic Memory Networks for
  Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf); данный проект является попыткой
  повторить и улучшить результаты, достигнутые авторами статьи. Также в реализации данного проекта были
  использованы свои идеи и идеи из следующих статей:

  - [Dynamic Memory Networks for Visual and Textual Question Answering](https://arxiv.org/pdf/1603.01417.pdf)
  - [Convolutional Sequence to Sequence Learning](https://s3.amazonaws.com/fairseq/papers/convolutional-sequence-to-sequence-learning.pdf)

  В связи с этим, называть данную архитектуру DMN — не вполне корректно; будем называть её DMN++.

## 1. Abstract

  Применять нейронные сети к задачам обработки естественного языка — изобретение не новое; существует
  множество различных архитектур, приспособленных к различным задачам — классификация текстов, определение
  настроения, определение авторства, расстановка тегов частей предложений и многие другие; однако многие из
  них могут быть сформулированы как пара "контекст — вопрос", например "Какое настроение у этого текста?" В
  связи с этим, сеть, способная обучаться отвечать на произвольные вопросы обладает большим спектром
  возможных применений.

  Dynamic Memory Network++ (DMN++) — сеть, состоящая из четырёх ключевых "блоков":

  - Блок вопроса: предобработка вопроса для передачи дальше;
  - Блок контекста: предобработка контекста и получение последовательности "фактов" для передачи дальше;
  - Блок эпизодической памяти: ключевой блок; совершает несколько проходов по фактам, используя механизм
    внимания, основанный на обработанном вопросе и состоянии памяти с прошлого прохода, обновляя созранённую
    память на каждом проходе;
  - Блок ответа: по финальному состоянию памяти сгенерировать последовательность слов ответа.

  Обучение сети и оценку качества работы будем производить на датасете
  [bAbI](https://research.fb.com/downloads/babi/)-en-10k.

### 2. Архитектура

  Рассмотрим по частям:

  #### I. Модуль вопроса

  Модуль обработки вопроса получает вопрос как последовательность номеров слов; если длина вопроса меньше
  максимальной длины, вопрос дополняется нужным числом повторений слова `PAD` в начале. После этого используя
  векторные представления слов, обучаемые вместе с сетью, последовательность слов конвертирууется в
  последовательность векторов, которые пропускаются через два GRU-слоя — в прямом направлении и в обратном.
  Полученные два вектора конкатенируются, и на выходе оказывается "сжатый" в один вектор `q` вопрос.

  #### II. Модуль контекста

  Основные отличия от DMN находятся именно здесь. В отличие от оригинальной статьи, текст подаётся на вход не
  сплошной последовательностью слов, а двумерной матрицей `C` размера `max_sentence_num X max_sentence_len`;
  в каждой строке матрицы находится ровно одно предложение. Опять же, до нужного размера матрица дополняется
  `PAD`.  Далее строится матрица `P` той же размерности, в которой каждому слову из `C` соответствует его
  номер в соответствующем предложении; таким образом, `P` выглядит как

    1 2 3 4 5 ...
    1 2 3 4 5 ...
    ...
    1 2 3 4 5 ...

  После этого, с помощью векторных представлений, слова контекста конвертируются в векторы, и номера слов, с
  помощью другой матрицы представлений, тоже конвертируются в векторы. После чего трёхмерные матрицы
  перемножаются поэлементно и складываются "вдоль" предложений — мы получаем одну двумерную матрицу размера
  `max_sentence_num X hidden_dim`. После этого двусторонняя GRU, возвращающая последовательности, и на выходе
  получается последовательность "фактов" `[f_1, ... f_n]`, соответствующих входным предложениям.

  #### III. Модуль эпизодической памяти

  Вектор памяти `m` инициализируется "сжатым" вопросом, полученным в пункте I; после чего совершается
  несколько (в конкретной реализации, 4) прохода по памяти:

  1. Высчитывается внимание: по каждому факту `f` строится вектор `a = (f, q, m, f*q, f*m, |f-q|, |f-m|)`; все
     операции — поэлементные; `( )` — конкатенация.

  2. Вектор `a` пропускается через два полносвязных слоя; второй имеет размерность 1, чтобы получить
     скалярный вес конкретного факта.

  3. После получения всей последовательности `[a_1, ..., a_n]`, вычисляется эпизод
     `e = a_1*f_1 + a_2*f_2 + ... + a_n*f_n`.

  4. Память обновляется: вектор `(m, e)` пропускается через полносвязный слой, после чего результат
     записывается в `m`.

  Реализовано всё было на Keras 2.0.0 и Python 3.6.0.

  #### IV. Модуль ответа.

  Полученная на шаге 3 память с помощью GRU и полносвязного слоя с `softmax`-активацией разворачивается в
  последовательность one-hot векторов, соответствующих выходным словам.


### 3. Результаты.

  Для оценки качества использовались три метрики:
  - `loss` (категориальная кросс-энтропия);
  - `elementwise_accuracy` — какая доля слов была предсказано правильно;
  - `answerwise_accuracy` — какая доля ответов была целиком предсказана правильно.

  Лучшие полученные результаты таковы:

            file       loss  elemwise_acc  answerwise_acc  | DMN_acc
    qa1_test.txt   0.000014      1.000000           1.000  |   1.000
    qa2_test.txt   0.017357      0.992667           0.956  |   0.982
    qa3_test.txt   0.101597      0.978667           0.872  |   0.952
    qa4_test.txt   0.000598      1.000000           1.000  |   1.000
    qa5_test.txt   0.005092      0.998667           0.992  |   0.993
    qa6_test.txt   0.002771      0.999333           0.996  |   1.000
    qa7_test.txt   0.009924      0.997667           0.986  |   0.969
    qa8_test.txt   0.005957      0.998333           0.992  |   0.965
    qa9_test.txt   0.000279      1.000000           1.000  |   1.000
    qa10_test.txt  0.004220      0.999000           0.994  |   0.975
    qa11_test.txt  0.000086      1.000000           1.000  |   0.999
    qa12_test.txt  0.000029      1.000000           1.000  |   1.000
    qa13_test.txt  0.000010      1.000000           1.000  |   0.998
    qa14_test.txt  0.001875      0.999333           0.996  |   1.000
    qa15_test.txt  0.000127      1.000000           1.000  |   1.000
    qa16_test.txt  0.174627      0.912333           0.474  |   0.994
    qa17_test.txt  0.086258      0.979000           0.874  |   0.596
    qa18_test.txt  0.027042      0.994000           0.964  |   0.953
    qa19_test.txt  0.049242      0.982667           0.904  |   0.345
    qa20_test.txt  0.000121      1.000000           1.000  |   1.000
    ----------------------------------------------------------------
    test_all.txt   0.018748      0.991667           0.952  |   0.936
    train_all.txt  0.005044      0.998333           0.990  |   -----
    valid_all.txt  0.025069      0.988667           0.936  |   -----


  Если пользоваться теми же границами прохождения задания, что и авторы оригинальной статьи, успешно пройдено
  16 заданий из 20; хуже всего сеть показывает себя на заданиях на

  - Basic induction: самый удивительный пункт; сеть из оригинальной статьи не испытывает никаких трудностей
    на этом задании. У меня нет объяснений.
  - Three supporting facts: задание само по себе несложное, однако длина контекстов явно сильно влияет на
    качество; истории по 150-160 предложений обрабатываются с трудом.
  - Positional reasoning: вероятно, последовательность перемещений в пространстве плохо переживает взвешенную
    сумму, используемую для обновления памяти.
  - Path finding: аналогично positional reasoning.

### 5. Заключение

  Результаты неплохи: средняя точность выше на 1.6%; в восьми заданиях из двадцати сеть показывает лучшее
  качество, чем оригинальная DMN, при этом перевес в пользу DMN++ достигает 55% на задании 19.

  Из того, что можно было бы улучшить, стоит отметить то, что стоит добавить возможность учитывать порядок
  фактов при обновлении памяти, например, добавив positional embeddings к этому этапу.

  Есть гипотеза, что GRU при подготовке фактов не принципиальна; возможно, хватит полносвязного слоя, а
  работу со сложными последовательностями фактво стоит переложить на модуль памяти; на проверку уже не хватит
  времени, впрочем.


### 6. Запуск

  Для проверки работы проекта достаточно выполнить `python3 server.py checkpoint` из папки `DMN`, после чего
  открыть в браузере `127.0.0.1:5000`. Интерфейс прост и примитивен:

  ![Интерфейс](https://raw.githubusercontent.com/Akiiino/Ask-Me-Anything/master/interface.png)

  Справа — одно текстовое поле для ввода вопроса, кнопка для отправки и кнопка, подгружающая случайный вопрос из
  тестовой выборки. Слева — визуализация весов, назначаемых предложениям механизмом внимания.
