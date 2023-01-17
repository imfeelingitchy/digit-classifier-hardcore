import gzip
import numpy as np

class Loader(object):

    def __init__(self, mini_batch_size = 10):
        self.mini_batch_size = mini_batch_size

    def load_train_images(self):
        with gzip.open("./data/train-images-idx3-ubyte.gz", "r") as f:
            f.read(16)
            data = f.read()
        return np.frombuffer(data, dtype=np.uint8).reshape(int(60000 / self.mini_batch_size), self.mini_batch_size, 28 * 28, 1) / 255

    def load_test_images(self):
        with gzip.open("./data/t10k-images-idx3-ubyte.gz", "r") as f:
            f.read(16)
            data = f.read()
        return np.frombuffer(data, dtype=np.uint8).reshape(10000, 28 * 28, 1) / 255

    def load_train_labels(self):
        def vectorized_label(x):
            a = [0] * 10
            a[x] = 1
            return a
        with gzip.open("./data/train-labels-idx1-ubyte.gz", "r") as f:
            f.read(8)
            data = f.read()
        int_labels = np.frombuffer(data, dtype=np.uint8)
        vectorized_labels = []
        for i in int_labels:
            vectorized_labels += vectorized_label(i)
        vectorized_labels = np.array(vectorized_labels)
        return vectorized_labels.reshape(int(60000 / self.mini_batch_size), self.mini_batch_size, 10, 1)

    def load_test_labels(self):
        with gzip.open("./data/t10k-labels-idx1-ubyte.gz", "r") as f:
            f.read(8)
            data = f.read()
        return np.frombuffer(data, dtype=np.uint8)

    def load_test_image_by_index(self, indx):
        with gzip.open("./data/t10k-images-idx3-ubyte.gz", "r") as f:
            f.read(16)
            f.read(indx * 28 * 28)
            data = f.read(28 * 28)
        return np.frombuffer(data, dtype=np.uint8).reshape(28, 28)