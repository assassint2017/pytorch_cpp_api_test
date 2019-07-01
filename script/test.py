import numpy as np

labels = np.loadtxt('../data/synset_words.txt', dtype=str, delimiter='\n')

print(type(labels))
print(labels)
print(labels.shape)