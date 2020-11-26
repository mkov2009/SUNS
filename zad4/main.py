import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import sklearn.decomposition as decomposition
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def contains(text, to_find):
    if to_find in text:
        return 1
    else:
        return 0


def remove_strings(data):
    data = data.drop(columns=['name', 'appid', 'developer', 'publisher', 'categories', 'genres', 'release_date', 'platforms', 'english' ])
    return data


def get_genre(genre):
    if contains(genre, "Strategy") == 1:
        return 1
    if contains(genre, "Casual") == 1:
        return 2
    if contains(genre, "Indie") == 1:
        return 3
    if contains(genre, "RPG") == 1:
        return 4
    if contains(genre, "Adventure") == 1:
        return 5
    if contains(genre, "Simulation") == 1:
        return 6
    if contains(genre, "Sexual Content") == 1:
        return 7
    if contains(genre, "Free to Play") == 1:
        return 8
    if contains(genre, "Sports") == 1:
        return 9
    return 10


def get_category(x):
    if x == 1:
        return 1
    elif x < 5:
        return 2
    elif x < 10:
        return 3
    else:
        return 4


def clear_data(df):
    # 1      /-1
    # 2-5    /-2
    # 5 - 10 /-3
    # > 10   /-4
    developer = df["developer"].value_counts()[df["developer"]]
    df["developerCategory"] = developer.values
    df["developerCategory"] = df["developerCategory"].map(lambda x: get_category(x))

    df["windows"] = df["platforms"].map(lambda x: contains(x, "windows"))
    df["mac"] = df["platforms"].map(lambda x: contains(x, "mac"))
    df["linux"] = df["platforms"].map(lambda x: contains(x, "linux"))

    df["single_player"] = df["categories"].map(lambda x: contains(x, "Single-player"))
    df["multi_player"] = df["categories"].map(lambda x: contains(x, "Multi-player"))

    data["genres"] = data["genres"].map(lambda x: get_genre(x))

    return df


def exploratory_data_analysis(data):
    fig = px.scatter(data, x="release_date", y="price")
    fig.update_traces(marker=dict(size=3))
    # fig.show()

    fig = px.scatter(data, x="price", y=data["positive_ratings"] - data["negative_ratings"])
    fig.update_traces(marker=dict(size=3))
    # fig.show()

    fig = px.scatter(data, x="release_date", y=data["positive_ratings"])
    fig.update_traces(marker=dict(size=3))
    # fig.show()

    data = clear_data(data)
    multiplayer = data.sort_values(by=['release_date']).groupby(['release_date']).sum(['multi_player'])
    fig = px.scatter(x=data['release_date'].unique(), y=multiplayer['multi_player'])
    fig.update_traces(marker=dict(size=3))
    # fig.show()


def plot_graphs(clustered_data):

    plt.figure(figsize=(10, 10))
    sns.countplot(x="cluster_id", hue="developerCategory", data=clustered_data)
    plt.show()

    sns.countplot(x="cluster_id", hue="linux", data=clustered_data)
    plt.show()

    sns.countplot(x="cluster_id", hue="mac", data=clustered_data)
    plt.show()

    sns.countplot(x="cluster_id", hue="owners", data=clustered_data)
    plt.show()

    grouped = clustered_data.groupby(["cluster_id"])
    sns.barplot(x=clustered_data["cluster_id"], y=grouped["average_playtime"].mean())
    plt.show()

    sns.boxplot(x=clustered_data["cluster_id"], y=clustered_data["difficult"])
    plt.show()


def cluster(data):
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    nClusters = 7

    clusters = MiniBatchKMeans(n_clusters=nClusters, verbose=True).fit(normalized_data)
    # clusters = cluster.DBSCAN(eps=7.2, min_samples=25).fit(normalized_data)

    data["cluster_id"] = clusters.labels_
    plot_graphs(data)

    # to_2Dgraph = pca_preparation(3, normalized_data, clusters.labels_)
    # graph_2d(to_2Dgraph,clusters.labels_)
    # to_graph = tsne_preparation(3, normalized_data, clusters.labels_)
    # graph_3d(to_graph,clusters.labels_)


def pca_preparation(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = decomposition.PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels


def tsne_preparation(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = TSNE(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i: names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels

    return df_matrix


def graph_2d(df, name='labels'):
    fig = px.scatter_3d(df, x='x', y='y', color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()


def graph_3d(df, name='labels'):
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=name, opacity=0.5)

    fig.update_traces(marker=dict(size=3))
    fig.show()


if __name__ == '__main__':
    df1 = pd.read_csv("data/steam.csv")
    df2 = pd.read_csv("data/steamspy_tag_data.csv")
    data = pd.merge(df1, df2, left_on='appid', right_on='appid', how='left')
    exploratory_data_analysis(data)
    data = clear_data(data)

    data = remove_strings(data)

    cluster(data)

    print("xoxo")
