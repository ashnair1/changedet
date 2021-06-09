import matplotlib.pyplot as plt


def histplot(xlist, xlabel):
    f = plt.figure()
    for i, j in zip(xlist, xlabel):
        plt.hist(i[:, :, 0].flatten(), bins=50, label=j)
    plt.legend()
    return f
