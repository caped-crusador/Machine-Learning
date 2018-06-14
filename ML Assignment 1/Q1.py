from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def show_examples():
    mnsit = input_data.read_data_sets("MNSIT_data/", one_hot=True)
    images = mnsit.train.images[0:50]
    print("examples 3 to 9", images[3:10])
    example_1 = plt.figure(1)
    plt.imshow(images[0].reshape([28, 28]), cmap="gray")
    example_1.show()
    example_2 = plt.figure(2)
    plt.imshow(images[1].reshape([28, 28]), cmap="gray")
    example_2.show()
    example_3 = plt.figure(3)
    plt.imshow(images[2].reshape([28, 28]), cmap="gray")
    example_3.show()
    example_4 = plt.figure(4)
    plt.imshow(images[3].reshape([28, 28]), cmap="gray")
    example_4.show()


show_examples()
