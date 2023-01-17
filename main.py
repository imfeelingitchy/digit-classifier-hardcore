import network
import mnist_loader
from render import display_number

loader = mnist_loader.Loader()
classifier = network.Network([28 * 28, 30, 10])
classifier.stochastic_gradient_descent(loader.load_train_images(), loader.load_train_labels(), 5, 3)

print("\nComplete. Accuracy: {} / 10000".format(classifier.evaluate_performance(loader.load_test_images(), loader.load_test_labels())))

while True:
    try:
        indx = int(input("\nEnter index of test image (0 - 9999): "))
    except:
        break
    img_data = loader.load_test_image_by_index(indx)
    display_number(img_data)
    classifier.predict(img_data)