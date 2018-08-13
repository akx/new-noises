import random


def generate_corpus_from_file(length, seps, filename="genres.txt"):
    lines = list(read_unique(filename))
    return generate_corpus_from_items(lines, length, seps)


def generate_corpus_from_items(items, length, seps):
    corpus = ""
    while len(corpus) <= length:
        corpus += "%s%s" % (random.choice(items), random.choice(seps))
    return corpus.strip()


def read_unique(filename):
    with open(filename, encoding="utf-8") as f:
        lines = [s.strip() for s in f.read().splitlines()]
        lines = {s for s in lines if s}
    return lines
