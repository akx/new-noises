from __future__ import print_function

import os
import re
import sys
import time

from itertools import count
import click
from keras.callbacks import LambdaCallback

import datautil
from char_rnn import CharRNNModel


@click.group()
def cli():
    pass


def sample_crn(stream, crn, seed, diversity, gen_length, avoid_reality=False, items=frozenset()):
    sampler = crn.sample(seed, diversity, length=gen_length)
    if avoid_reality:
        data = ''.join(sampler)
        for item in re.split(r'\s+', data):
            if item not in items:
                stream.write(item)
                stream.write("\n")
    else:
        last = None
        for chunk in sampler:
            if chunk.isspace() and last == chunk:
                continue
            stream.write(chunk)
            last = chunk
            stream.flush()
        stream.write("\n")


@cli.command()
@click.option("--input-file", default="genres.txt")
@click.option("--length", type=int, default=100_000)
@click.option("--sample-size", type=int, default=15)
@click.option("--gen-length", type=int, default=150)
@click.option("--units", type=int, default=128)
@click.option("--avoid-reality/--no-avoid-reality", default=False)
def train(input_file, length, sample_size, gen_length, units, avoid_reality):
    input_file_name = os.path.splitext(os.path.basename(input_file))[0]

    tag = f"cx_{input_file_name}_{int(time.time())}"
    print(f"Tag: {tag}")

    items = list(datautil.read_unique(input_file))

    text = datautil.generate_corpus_from_items(
        items=items,
        length=length,
        seps=(" ", " ", " ", "\n", "\n"),
    )

    crn = CharRNNModel(text, sample_length=sample_size)
    x, y = crn.generate_training_data(text)
    crn.build_model(units=units, activation="softmax")

    def on_epoch_end(epoch, logs):
        filename = f"{tag}_{epoch}.hdf5"
        crn.save(filename)
        print(f"----- {filename} - generating text after epoch {epoch}")
        for seed in [crn.generate_sample_seed(text)]:
            p_seed = seed.replace('\n', '·')
            for diversity in [0.2, 0.5, 1.0, 1.2]:
                print(f'----- Generating with diversity {diversity}, seed "{p_seed}"')
                sample_crn(
                    stream=sys.stdout,
                    crn=crn,
                    seed=seed,
                    diversity=diversity,
                    gen_length=gen_length,
                    avoid_reality=avoid_reality,
                    items=items,
                )

    crn.model.fit(
        x,
        y,
        batch_size=32,
        epochs=60,
        callbacks=[(LambdaCallback(on_epoch_end=on_epoch_end))],
    )


@cli.command()
@click.option("--model-file", default="char_rnn_genres_5.hdf5")
@click.option("--seed-mode", type=click.Choice(['empty', 'random']), default='random')
@click.option("--gen-length", type=int, default=150)
@click.option("--diversity", type=float, default=1.2)
@click.option("--avoid-reality/--no-avoid-reality", default=False)
@click.option("--reality-file")
def sample(model_file, seed_mode, gen_length, diversity, avoid_reality, reality_file):
    if avoid_reality:
        items = list(datautil.read_unique(reality_file))
    else:
        items = []

    crn = CharRNNModel.load(model_file)

    for x in count():
        if seed_mode == 'random':
            seed = crn.generate_random_seed()
        else:
            seed = " " * crn.sample_length

        sample_crn(
            stream=sys.stdout,
            crn=crn,
            seed=seed,
            diversity=diversity,
            gen_length=gen_length,
            avoid_reality=avoid_reality,
            items=items,
        )


if __name__ == '__main__':
    cli()
