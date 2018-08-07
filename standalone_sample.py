import sys
from itertools import count

from char_rnn import CharRNNModel

crn = CharRNNModel.load("char_rnn_genres_5.hdf5")

for x in count():
    seed = crn.generate_random_seed() if x % 2 else " " * crn.sample_length
    for chunk in crn.sample(seed, 1.2):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    print(f"\n")
