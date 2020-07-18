import numpy as np


def random_arr():
    data = np.random.randn(2, 3)
    print("평균 0, 표준편차 1의 가우시안 표준정규분포를 따르는 난수\n")
    print("{0}\n".format(data))
    data = np.random.rand(2, 3)
    print("0 ~ 1 균일분포 표준정규분포를 따르는 난수\n")
    print("{0}\n".format(data))


def ndArr():
    data = [6, 7.5, 8, 0, 1]
    arr1 = np.array(data)
    print(arr1)


def numpyArange():
    arr = np.arange(15).reshape((3, 5))
    print(arr)
    print(arr.T)


def getmeshgrid():
    points = np.arange(-5, 5, 0.01)
    xs, ys = np.meshgrid(points, points)
    z = np.sqrt(xs ** 2 + ys ** 2)
    import matplotlib.pyplot as plt
    plt.imshow(z, cmap=plt.cm.gray)
    plt.colorbar()
    plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
    plt.show()


def linear_algebra():
    x = np.array([[1., 2., 3.], [4., 5., 6.]])
    y = np.array([[6., 23.], [-1, 7], [8, 9]])
    print(x.dot(y))


def stairs():
    import random
    position = 0
    walk = [position]
    steps = 1000
    for i in range(steps):
        step = 1 if random.randint(0, 1) else -1
        position += step
        walk.append(position)

    import matplotlib.pyplot as plt
    plt.plot(walk[:100])
    plt.show()


def dot_product():
    A = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
    it = np.nditer(A, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        print(idx)
        it.iternext()


dot_product()
