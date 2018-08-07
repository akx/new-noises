from __future__ import print_function

import json
import logging
import random

import numpy as np
from keras.engine.saving import load_model
from keras.layers import Activation, Dense, LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop


class cached_property:
    def __init__(self, func, name=None):
        self.func = func
        self.__doc__ = getattr(func, "__doc__")
        self.name = name or func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


class CharRNNModel:
    log = logging.getLogger(__name__)

    def __init__(self, chars, sample_length=15):
        self.chars = sorted(list(set(chars)))
        self.sample_length = sample_length

    @cached_property
    def char_indices(self):
        return dict((c, i) for i, c in enumerate(self.chars))

    @cached_property
    def indices_char(self):
        return dict((i, c) for i, c in enumerate(self.chars))

    def build_model(self, units=128, activation="softmax", optimizer=RMSprop(lr=0.015)):
        model = Sequential()
        model.add(LSTM(units, input_shape=(self.sample_length, len(self.chars))))
        model.add(Dense(len(self.chars)))
        model.add(Activation(activation))
        model.compile(loss="categorical_crossentropy", optimizer=optimizer)
        self.model = model

    def generate_training_data(self, text, step=None):
        maxlen = self.sample_length
        if step is None or step <= 0:
            step = maxlen // 4
        sentences = []
        next_chars = []
        for i in range(0, len(text) - maxlen, step):
            sentences.append(text[i : i + maxlen])
            next_chars.append(text[i + maxlen])

        self.log.info(
            f"vectorizing {len(sentences)} sentences from {len(text)} characters of data"
        )

        x = np.zeros((len(sentences), maxlen, len(self.chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(self.chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1

        return (x, y)

    def sample(self, sentence, diversity: float, length: int = 400):
        if len(sentence) != self.sample_length:
            raise ValueError(f"seed must be {self.sample_length} characters long")
        if not (set(sentence) <= set(self.chars)):
            raise ValueError(f"seed contains invalid characters")

        generated = sentence

        for i in range(length):
            x_pred = np.zeros((1, self.sample_length, len(self.chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, self.char_indices[char]] = 1.

            preds = self.model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = self.indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            yield next_char

    def generate_sample_seed(self, text):
        start_index = random.randint(0, len(text) - self.sample_length - 1)
        return text[start_index : start_index + self.sample_length]

    def generate_random_seed(self):
        return "".join(random.choice(self.chars) for x in range(self.sample_length))

    def save(self, filepath):
        import h5py

        with h5py.File(filepath, mode="w") as f:
            self.model.save(f)
            f.attrs["char_rnn_init_args"] = json.dumps(
                {"chars": "".join(self.chars), "sample_length": self.sample_length}
            )

    @classmethod
    def load(cls, filepath):
        import h5py

        with h5py.File(filepath, mode="r") as f:
            init_args = json.loads(f.attrs["char_rnn_init_args"])
            model = cls(**init_args)
            model.model = load_model(filepath)
        return model
