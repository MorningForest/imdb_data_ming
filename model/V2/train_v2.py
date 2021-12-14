# -*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/13 14:46
import sys
import argparse
sys.path.append("../")

import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl
from fastNLP import DataSet, Instance, Trainer, Tester, DataSetIter
from fastNLP.core import AccuracyMetric, CrossEntropyLoss, callback, cache_results

from pipe import load_data
from model import Model

# 设置中文字体
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.path = '../../data/movie_metadata.csv'
args.embed_size = 300
args.hidden_size = 200
args.lr = 4e-3
args.n_epochs = 10
args.batch_size = 32
args.n_clusters = 3
args.save_path = r"data.pkl"
args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@cache_results(args.save_path, _refresh=False)
def get_data():
    return load_data(args.path)

df_matrix, vocab, ds_train, ds_test = get_data()

class SaveCallback(callback.Callback):
    def __init__(self):
        super(SaveCallback, self).__init__()
        self.filepath = r"model.pkl"

    def on_valid_end(self, eval_result, metric_key, optimizer, is_better_eval):
        if is_better_eval:  # 比上一次测试结果高
            model = self.model.to("cpu")
            model.feat_mat = model.feat_mat.to("cpu")
            torch.save(model, self.filepath)
            self.model.to(self.trainer._model_device)
            self.model.feat_mat = self.model.feat_mat.to(self.trainer._model_device)


def train_cls_model():
    # 训练一个分类模型
    model = Model(len(vocab), embed_size=args.embed_size, hidden_size=args.hidden_size, feat_mat=df_matrix)
    model.feat_mat = model.feat_mat.to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    trainer = Trainer(ds_train, model=model, metrics=AccuracyMetric(target="label"), loss=CrossEntropyLoss(target="label"),
                    dev_data=ds_test, batch_size=args.batch_size, device=args.device, callbacks=[SaveCallback()], optimizer=optim,
                    n_epochs=args.n_epochs)
    trainer.train()

    # tester = Tester(ds_test, model=model, metrics=AccuracyMetric(target="label"), device=torch.device("cuda"))
    # tester.test()

def cluster():
    device = torch.device("cuda")
    model = torch.load("model.pkl").to(device)
    model.feat_mat = model.feat_mat.to(device)

    dsiter = DataSetIter(ds_train.concat(ds_test))

    new_ds = DataSet()
    for batch_x, _ in dsiter:
        out = model.forward(batch_x['input_ids'].to(device), batch_x['ins_id'].to(device))
        new_ds.append(Instance(ins_id=batch_x['ins_id'], feat=out['mix_feat'][0].to("cpu").detach().numpy()))
    # print(new_ds)

    inp = np.array(new_ds.get_field("feat").content)
    from sklearn.cluster import KMeans

    y_pred = KMeans(n_clusters=args.n_clusters, random_state=9).fit_predict(inp)
    cluster_label = pd.DataFrame(y_pred, columns=['cluster'])

    # 多维数据可视化
    tsne_out = TSNE(n_components=2).fit_transform(inp)
    pos = pd.DataFrame(tsne_out, columns=['X', 'Y'])
    #  数据可视化
    plt.figure()
    pos.plot(kind='scatter', x='X', y='Y', color='green')
    plt.show()
    #  聚类后的数据可视化
    pos['cluster'] = cluster_label
    plt.figure()
    cor_list = ['blue', 'green', 'black', 'red']
    ax = pos[pos['cluster'] == 0].plot(kind='scatter', x='X', y='Y', color='blue', label='0')

    for k in range(1, args.n_clusters):
        pos[pos['cluster'] == k].plot(kind='scatter', x='X', y='Y', color=cor_list[k % 4], label='%d' % k, ax=ax)
    plt.show()

if __name__ == '__main__':
    train_cls_model()
    cluster()