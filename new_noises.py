from __future__ import print_function

import sys
import time

from keras.callbacks import LambdaCallback

import datautil
from char_rnn import CharRNNModel

text = datautil.generate_corpus(100_000, seps=(" ", " ", " ", "\n", "\n"))

crn = CharRNNModel(text, 15)
x, y = crn.generate_training_data(text)
crn.build_model(128, "softmax")

tag = f"cx_{int(time.time())}"


def on_epoch_end(epoch, logs):
    filename = f"{tag}_{epoch}.hdf5"
    crn.save(filename)
    print(f"----- {filename} - generating text after epoch {epoch}")
    for seed in [crn.generate_sample_seed(text)]:
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print(f'----- Generating with diversity {diversity}, seed "{seed}"')
            for chunk in crn.sample(seed, diversity):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            print(f"\n")


crn.model.fit(
    x,
    y,
    batch_size=32,
    epochs=60,
    callbacks=[(LambdaCallback(on_epoch_end=on_epoch_end))],
)
