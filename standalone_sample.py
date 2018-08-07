import sys

from char_rnn import CharRNNModel

crn = CharRNNModel.load("char_rnn_genres_6.hdf5")

for chunk in crn.sample(" " * crn.sample_length, 1.3):
    sys.stdout.write(chunk)
    sys.stdout.flush()
print(f"\n")
