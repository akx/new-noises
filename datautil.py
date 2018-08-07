import random


def generate_corpus(length, seps):
    with open("genres.txt", encoding="utf-8") as f:
        lines = [s.strip() for s in f.read().splitlines()]
        lines = [s for s in lines if s]
    corpus = ""
    while len(corpus) <= length:
        corpus += "%s%s" % (random.choice(lines), random.choice(seps))
    return corpus.strip()
