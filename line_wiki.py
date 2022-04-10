
import numpy as np
from classify import read_node_label, Classifier
from line import LINE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import time
from argsss import parameter_parser
import  warnings
warnings.filterwarnings('ignore')

def evaluate_embeddings(embeddings, args):
    X, Y = read_node_label(args.file_label) #获取真实标签
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,args):
    data_label = args.file_label
    X, Y = read_node_label(data_label)

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)

    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    args = parameter_parser()
    newdata = 'newdata/'+ args.new_file
    olddata = 'data/'+args.old_file
    # G = nx.read_edgelist(newdata, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    G = nx.read_edgelist(olddata, create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    time1 = time.time()

    time2 = time.time()
    print('：', time2 - time1)
    model = LINE(G, embedding_size=128, order='all')
    for i in np.arange(1,7):
        model.train(batch_size=1024, epochs=4, verbose=2)
        print('batch_it', model.batch_it)
        print('hist', model.hist)
        embeddings = model.get_embeddings()

        evaluate_embeddings(embeddings,args)

    time2 = time.time()
    print('时间为：', time2 - time1)
    # plot_embeddings(embeddings, args)