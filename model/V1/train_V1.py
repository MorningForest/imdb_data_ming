import sys
sys.path.append('../')
from V1.model.correlation import Apriori
from V1.model.base_info import data_info
from V1.model.cluster import kmeansModel, analyse_cluster, LevelCluster, DBSCANCluster
from V1.process.pipe import MVLoader, ClusterPipe, RegrePipe, CorrelationPipe
from V1.model.regression import CnnReg, RegMetric, RegLSTM

from fastNLP import Trainer, L1Loss, Callback
from fastNLP import SequentialSampler, DataSetIter
import argparse
import torch
import fitlog

## fitlog日志
fitlog.set_log_dir("logs/")

parse = argparse.ArgumentParser()
parse.add_argument('--method', required=True, type=str)
args = parse.parse_args()
args.embed_size = 64
args.lr = 0.01
args.batch_size = 32
args.n_epoch = 20

fitlog.add_hyper(args)
path = r'../../data/movie_metadata.csv'
log_dir = r'./log/log'
paths = {'train': '../../data/train.csv',
         'dev': '../../data/dev.csv',
         'test': '../../data/test.csv'}

if args.method not in ['rel', 'cluster', 'cls']:
    IOError('输入参数有问题， 请输入rel, cluster, cls')


# demo = False
# if demo:
#     ds, vocab = RegrePipe(demo).process_from_file(paths)
# else:
#     ds, vocab = RegrePipe(demo).process_from_file(paths)
# train = ds.get_dataset('train')
# dev = ds.get_dataset('dev')
# test = ds.get_dataset('test')
# model = CnnReg(len(vocab), args.embed_size)
# opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
# trainer = Trainer(train, model, loss=MSELoss(target='imdb_score'), optimizer=opt,
#         device=None, batch_size=args.batch_size, metrics=AccuracyMetric(target='imdb_score'),
#         print_every=1, dev_batch_size=args.batch_size, n_epochs=args.n_epoch,
#         dev_data=dev)
#
# trainer.train()
# ds, df = MVLoader()._load(path)
#
# data_info(log_dir).describe(df)
# inp = ClusterPipe().process(df)
# km = kmeansModel(log_dir)
# act = analyse_cluster(log_dir)
# best_k, best_kmeans, cluster_label_k, score_list = km.train(inp, 2, 5)
# act.describe(df, cluster_label_k, best_k)

# ds = ds.get_dataset('train')
# # ds.apply_field(lambda x:x.split('|'), field_name='genres', new_field_name='genres')
# from fastNLP import DataSet, Instance
# ins = []
# import pandas as pd
# for item in ds:
#     res = []
#     res.append(item['color'])
#     res.append(item['director_name'])
#     res.extend(item['genres'].split('|'))
#     res.append(item['actor_1_name'])
#     res.append(item['actor_3_name'])
#     res.append(item['actor_2_name'])
#     res.append(item['language'])
#     res.append(item['country'])
#     ins.append(res)
# ap = Apriori(ins, './log/log')
# freq, sp = ap.apriori(min_support=0.01)
# print(freq)
# with open('res.txt', 'a+', encoding="utf8") as fp:
#     for item in freq:
#         for v in item:
#             for k in v:
#                 fp.write(str(v)+" ")
#             fp.write("\n")
# rules = ap.association_rules(freq, sp, 0.1)
# print(rules)

if args.method == 'rel':
    min_support = 0.01 #支持度
    conf = 0.1 #置信度
    ## 关联的列
    cols_name = ['color', 'director_name', 'genres', 'actor_1_name',
                 'actor_2_name', 'actor_3_name', 'language', 'country',
                 'content_rating']
    demo = False
    _, df = MVLoader(demo)._load(path)
    data_info(log_dir).describe(df)
    inp = CorrelationPipe().process(df, cols_name)
    ap = Apriori(inp, log_dir)
    freq, sp = ap.apriori(min_support=0.01)
    # rules = ap.asscoiation_rules(freq, sp, conf)
    print(freq)
if args.method == 'cluster':
    n_clus_low, n_clus_high = 2, 3
    demo = False
    ds, df = MVLoader(demo)._load(path)

    data_info(log_dir).describe(df)
    inp = ClusterPipe().process(df)
    act = analyse_cluster(log_dir)
    # from sklearn.manifold import TSNE
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # cols_num = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes',
    #             'actor_1_facebook_likes', 'gross', 'cast_total_facebook_likes',
    #             'facenumber_in_poster', 'num_user_for_reviews', 'budget',
    #             'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']
    # tsne = TSNE(n_components=2)
    # inp = tsne.fit_transform(df[cols_num])
    # inp = pd.DataFrame(inp, columns=['X', 'Y'])
    # plt.figure()
    # inp.plot(kind='scatter', x='X', y='Y', color='green')
    # plt.show()
    ## Kmeans聚类 --->效果最好
    km = kmeansModel(log_dir)
    best_k, best_kmeans, cluster_label_k, score_list = km.train(inp, n_clus_low, n_clus_high)
    act.describe(df, cluster_label_k, best_k)
    ## 层次聚类
    linkage_methods = ['single', 'complete', 'average', 'ward'] ##距离方式
    lc = LevelCluster()
    for link in linkage_methods:
        cluster_label = lc.train(link, inp, 6)
        act.describe(df, cluster_label, n_clus_high)
    ##密度聚类
    eps = 3
    min_sample = 100
    dsl = DBSCANCluster()
    cluster_label = dsl.train(inp, eps, min_sample)
    # print(cluster_label)
    act.describe(df, cluster_label, len(set(cluster_label)))


if args.method == 'cls':
    demo = False
    if demo:
        ds, vocab = RegrePipe().process_from_file(paths, demo)
    else:
        ds, vocab = RegrePipe().process_from_file(paths)
    train = ds.get_dataset('train')
    dev = ds.get_dataset('dev')
    test = ds.get_dataset('test')
    callback = []

    class SaveCallback(Callback):
        def __init__(self):
            super(SaveCallback, self).__init__()
        def on_valid_begin(self):
            print("save model!!!")
            self.model.cpu()
            torch.save(self.model, 'tmp.pt')
            self.model.to(self.trainer._model_device)
    callback.append(SaveCallback())

    model = CnnReg(len(vocab), args.embed_size)
    # model = RegLSTM(len(vocab), args.embed_size)
    # model = torch.load('tmp.pt')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    trainer = Trainer(train, model, loss=L1Loss(target='imdb_score'), optimizer=opt,
            device=None, batch_size=args.batch_size, metrics=RegMetric(target='imdb_score'),
            print_every=1, dev_batch_size=args.batch_size, n_epochs=args.n_epoch,
            dev_data=dev, callbacks=callback)

    trainer.train()
    ##test集合上的情况
    batch = DataSetIter(batch_size=args.batch_size, sampler=SequentialSampler(), dataset=test)
    y_pred, y_true = [], []
    for batch_x, batch_y in batch:
        out = model.forward(batch_x['input_ids'])
        y_pred.append(out['pred'])
        y_true.append(batch_y['imdb_score'])
    y_pred = torch.cat(y_pred).tolist()
    y_true = torch.cat(y_true).tolist()

    import matplotlib.pyplot as plt
    plt.subplot(2, 2, 1)
    plt.plot(y_pred, label='y_pred', color='green')
    plt.legend(loc='upper right')
    plt.xlabel('电影id')
    plt.ylabel('imdb_score')
    plt.subplot(2, 2, 2)
    plt.plot(y_true, label='y_true', color='red')
    plt.legend(loc='upper right')
    plt.xlabel('电影id')
    plt.ylabel('imdb_score')
    plt.subplot(2, 2, 3)
    plt.plot(y_pred, label='y_pred', color='green')
    plt.plot(y_true, label='y_true', color='red')
    plt.legend(loc='upper right')
    plt.xlabel('电影id')
    plt.ylabel('imdb_score')

    plt.show()

