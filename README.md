# Ask Me Anything
## Попов Никита Сергеевич, группа 151-2

### 1. Цели:

  Повторить и, возможно, улучшить результат, описанный в статье [Ask Me Anything:
Dynamic Memory Networks for Natural Language Processing](https://arxiv.org/pdf/1506.07285.pdf), то есть построить модель нейронной сети, приспособленной для обработки текста и извлечения из него смысловых связей.

### 2. Задачи проекта

  1. Изучить общую теорию нейронных сетей:
    1. Различные архитектуры: перцептрон, рекуррентные сети, GRU-сети,
     LSTM-сети;
    2. Различные методы обучения;
    3. Способы работы с текстом в нейронных сетях.
  2. Понять результаты и методы их достижения, описанные в оригинальной статье, и то, почему они работают.
  3. Применить теоретические знания на практике и воспроизвести модель, описанную в статье; добиться схожих
     результатов.
  4. Исследовать способы улучшить качество модели — например, скорость обучения или качество работы с текстом.

### 3. Актуальность задачи

  В последнее время нейросети, в частности, глубокое обучение и рекуррентные сети всё активнее
  развиваются; в соответствии с этим растёт и область их применения. Сети, способные распознавать и извлекать
  какой-то смысл из текста на естественном языке, таким образом, является лишь логическим продолжением череды разработок по этим темам в последние годы. Понятно, что распознавание естественной речи может быть применено много где — от относительно простых и более прикладных ботов для колл-центров и прочих электронных консультантов, до более сложных задач анализа текстов; также полезны могут оказаться теоретические результаты, на основе которых могут быть написаны новые статьи, углубляющие и расширающие область применения таких сетей ещё дальше.

### 4. Обзор существующих решений

  Преимущественно в качестве "эталонного" решения будет использована указаннвя выше статья, как основной
  источник информации по конкретной архитектуре сети.


### 5. Используемые решения

  * Основной язык разработки — Python;
  * Библиотека для матричных и не только вычислений — numpy, как де-факто стандарт;
  * Библиотеки, используемые для моделирования сетей и выполнения вычислений — Theano и TensorFlow, как одни
    из самых известных и использумых;
  * Библиотеки для высокоуровневой работы с сетями — Keras и Lasagne; они позволяют упростить построение
    архитектуры сети и проведение экспериментов.

### 6. План работы
  0. 3.01.2017 — deadline лабораторной работы №3: конкурс по распознаванию иероглифов.
  1. 2.02.2017 — завершена подготовка и лабораторные работы; переход к работе со статьями, в том числе
  оригинальной.
  2. 3.03.2017 — завершено обсуждение тематических статей; далее ведётся только работа над практической
     реализацией.
  3. 3.06.2017 — последний milestone: сеть завершена, обучена и готова к демонстрации.