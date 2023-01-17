import network
import mnist_loader

loader = mnist_loader.Loader()
classifier = network.Network([28 * 28, 30, 10])
classifier.stochastic_gradient_descent(loader.load_train_images(), loader.load_train_labels(), 5, 3)

print("Complete. Accuracy: {} / 1000".format(classifier.evaluate_performance(loader.load_test_images(), loader.load_test_labels())))