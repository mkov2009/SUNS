import pandas as pd
import matplotlib.pyplot as plt
import sklearn.tree as tree
import seaborn as sns
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from os import system
import tensorflow as tf
import tensorflow.keras as kr
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression


def data_clear(data):
    data = data.dropna(how="any", axis=0)
    data = data.drop(columns=['objid', 'rerun', 'specobjid', 'fiberid', 'mjd', 'run'])

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


def plot_confusion_matrix(test_y, pred_y):
    conf_m = tf.math.confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_m, annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


def print_errors(test_y, y_pred):
    print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(test_y, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, y_pred)))


def tree_classifier(train, test):

    train_y = pd.get_dummies(train['class'])
    train_x = train.drop(columns='class')

    test_y = test['class']
    test_x = test.drop(columns='class')

    clf = tree.DecisionTreeClassifier(min_samples_leaf=8)
    clf.fit(train_x, train_y)
    pred_y = np.argmax(clf.predict(test_x), axis=1)
    score = clf.score(test_x, test_y)
    print("Model score: ", score)

    # Tree visualization
    dotfile = open("D:/SUNS/Z3/dtree1.dot", 'w')
    tree.export_graphviz(clf, out_file=dotfile, feature_names=train_x.columns)
    dotfile.close()
    system("dot -Tpng D:/SUNS/Z3/dtree1.dot -o D:/SUNS/Z3/dtree1.png")

    # Confusion matrix
    plot_confusion_matrix(test_y, pred_y)


def forest_classifier(train, test):
    train_y = train['class']
    train_x = train.drop(columns='class')

    test_y = test['class']
    test_x = test.drop(columns='class')

    sampler = RandomUnderSampler(random_state=42)
    train_x, train_y = sampler.fit_sample(train_x, train_y)

    forest_clf = RandomForestClassifier(max_depth=10, random_state=42, n_estimators=1)
    forest_clf.fit(train_x, train_y)

    pred_y = forest_clf.predict(test_x)
    plot_confusion_matrix(test_y, pred_y)


def nn(df_train, df_test):

    train, validate = np.split(df_train.sample(frac=1), [int(.8 * len(df_train))])

    train_y = pd.get_dummies(train['class'])
    train_x = train.drop(columns='class')

    val_y = pd.get_dummies(validate['class'])
    val_x = validate.drop(columns='class')

    test_y = df_test['class']
    test_x = df_test.drop(columns='class')

    model = kr.Sequential()

    # Neural network
    model.add(kr.layers.Dense(100, input_dim=11, activation="sigmoid"))
    model.add(kr.layers.Dense(50, activation="sigmoid"))
    model.add(kr.layers.Dense(3, activation="sigmoid"))
    model.summary()

    optimizer = kr.optimizers.Adam(learning_rate=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    early_stop = kr.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    training = model.fit(train_x, train_y, epochs=1000, validation_data=(val_x, val_y), callbacks=[early_stop])

    predicted = np.argmax(model.predict(test_x), axis=1)

    # Confusion matrix
    plot_confusion_matrix(test_y, predicted)

    # Summarize history for accuracy
    plt.plot(training.history['accuracy'])
    plt.plot(training.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Summarize history for loss
    plt.plot(training.history['loss'])
    plt.plot(training.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def forest_regression(train, test):
    train_y = train[['x_coord', 'y_coord', 'z_coord']]
    train_x = train.drop(columns=['x_coord', 'y_coord', 'z_coord'])

    test_y = test[['x_coord', 'y_coord', 'z_coord']]
    test_x = test.drop(columns=['x_coord', 'y_coord', 'z_coord'])

    regr = RandomForestRegressor(n_estimators=20, random_state=0)
    regr.fit(train_x, train_y)
    y_pred = regr.predict(test_x)

    print("Score: ", regr.score(test_x, test_y))
    print_errors(test_y, y_pred)


def neighbors_regression(train, test):
    train_y = train[['x_coord', 'y_coord', 'z_coord']]
    train_x = train.drop(columns=['x_coord', 'y_coord', 'z_coord'])

    test_y = test[['x_coord', 'y_coord', 'z_coord']]
    test_x = test.drop(columns=['x_coord', 'y_coord', 'z_coord'])

    regr = KNeighborsRegressor()
    regr.fit(train_x, train_y)
    y_pred = regr.predict(test_x)

    print("Score: ", regr.score(test_x, test_y))
    print_errors(test_y, y_pred)


def stacking_classifier(train, test):
    train_y = train['class']
    train_x = train.drop(columns='class')

    test_y = test['class']
    test_x = test.drop(columns='class')

    estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
                  ('svr', make_pipeline(StandardScaler(), LinearSVC(random_state=42)))]
    clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)

    print("Score: ", clf.score(test_x, test_y))
    plot_confusion_matrix(test_y, y_pred)


if __name__ == '__main__':
    data_test = pd.read_csv("test.csv")
    data_train = pd.read_csv("train.csv")

    data_test = data_clear(data_test)
    data_train = data_clear(data_train)
    # plot_graphs(data_train)
    # tree_classifier(data_train, data_test)
    # nn(data_train, data_test)
    # forest_regression(data_train, data_test)
    # neighbors_regression(data_train, data_test)
    stacking_classifier(data_train, data_test)
