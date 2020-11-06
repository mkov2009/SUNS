import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sklearn.tree as tree
import seaborn as sns


def data_clear(data):
    data = data.dropna(how="any", axis=0)
    data = data.drop(columns=['objid', 'rerun', 'specobjid'])

    data = data.drop(columns=['fiberid', 'mjd'])

    data['class'] = data['class'].map({'STAR': 0, 'GALAXY': 1, 'QSO': 2})

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


def tree_classifier(train, test):
    train = train.drop(columns=['x_coord', 'y_coord', 'z_coord'])
    test = test.drop(columns=['x_coord', 'y_coord', 'z_coord'])

    train_y = pd.get_dummies(train['class'])
    train_x = train.drop(columns='class')

    test_y = test['class']
    test_x = test.drop(columns='class')

    clf = tree.DecisionTreeClassifier(min_samples_leaf=10)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    score = clf.score(train_x, train_y)

    print("Model score: ", score)

    # Confusion matrix
    conf_m = sklearn.metrics.confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_m, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


if __name__ == '__main__':
    data_test = pd.read_csv("test.csv")
    data_train = pd.read_csv("train.csv")

    data_test = data_clear(data_test)
    data_train = data_clear(data_train)
    # plot_graphs(data_train)
    tree_classifier(data_train, data_test)
