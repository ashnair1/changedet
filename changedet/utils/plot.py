import matplotlib.pyplot as plt


def histplot(xlist, xlabel, bins=50):
    """Plot multiple histograms in the same figure

    Args:
        xlist (list[]): Sequence
        xlabel (list[str]): Sequence label
        bins (int, optional): Histogram bins. Defaults to 50.

    Returns:
        matplotlib.pyplot.figure: Figure with histograms
    """
    f = plt.figure()
    for i, j in zip(xlist, xlabel):
        plt.hist(i[:, :, 0].flatten(), bins=bins, label=j)
    plt.legend()
    return f
