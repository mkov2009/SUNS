import pandas as pd
import matplotlib.pyplot as plt


def data_clear(data):
    data = data.dropna(how="any", axis=0)
    data = data.drop(columns=['objid', 'rerun', 'specobjid'])

    data = data.drop(columns=['fiberid', 'mjd'])

    data['class'] = data['class'].map({'STAR': 1, 'GALAXY': 2, 'QSO': 3})

    return data


def plot_graphs(df):
    plt.plot(df['u'])
    plt.title('u')
    plt.show()
    plt.plot(df['g'])
    plt.title('g')
    plt.show()
    plt.plot(df['r'])
    plt.title('r')
    plt.show()
    plt.plot(df['i'])
    plt.title('i')
    plt.show()
    plt.plot(df['z'])
    plt.title('z')
    plt.show()
    plt.plot(df['run'])
    plt.title('run')
    plt.show()
    plt.plot(df['camcol'])
    plt.title('camcol')
    plt.show()
    plt.plot(df['field'])
    plt.title('field')
    plt.show()
    plt.plot(df['class'])
    plt.title('class')
    plt.show()
    plt.plot(df['plate'])
    plt.title('plate')
    plt.show()
    plt.plot(df['x_coord'])
    plt.title('x_coord')
    plt.show()
    plt.plot(df['y_coord'])
    plt.title('y_coord')
    plt.show()
    plt.plot(df['z_coord'])
    plt.title('z_coord')
    plt.show()


if __name__ == '__main__':
    data_test = pd.read_csv("test.csv")
    data_train = pd.read_csv("train.csv")

    data_test = data_clear(data_test)
    data_train = data_clear(data_train)
    # plot_graphs(data_train)
