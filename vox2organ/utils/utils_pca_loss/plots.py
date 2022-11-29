import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt



def plot_bar(data, y_label='', x_label='', title='', save_path=None, filename='test.pdf'):
    if save_path == None:
        raise NotImplementedError
    y_pos = np.arange(len(data))
    plt.bar(y_pos, data, align='center', alpha=0.5)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)

    plt.savefig(save_path + "/" + filename)  