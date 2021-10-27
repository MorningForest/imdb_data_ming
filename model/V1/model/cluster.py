from sklearn.cluster._kmeans import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from .utils import load_logger

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

class kmeansModel:
    def __init__(self, log_dir):
        self.logger = load_logger(log_dir)
    def forward(self, inp, n_cluster):
        model = KMeans(n_clusters=n_cluster)
        label_tmp = model.fit_predict(inp)
        silhouette_tmp = silhouette_score(inp, label_tmp) ##计算轮廓系数
        return silhouette_tmp, label_tmp, model
    def train(self, inp, cluster_low, cluster_high):
        silhouette_int = -1
        best_k = cluster_low
        best_kmeans, cluster_label_k = None, None
        score_list = list()
        for n_cluster in range(cluster_low, cluster_high):
            silhouette_tmp, label_tmp, model = self.forward(inp, n_cluster)
            if silhouette_tmp > silhouette_int:
                best_k = n_cluster
                silhouette_int = silhouette_tmp
                best_kmeans = model
                cluster_label_k = label_tmp
            score_list.append([n_cluster, silhouette_tmp])
            self.logger.info("n_cluster: {} ,silhouette score: {}".format(n_cluster, silhouette_tmp))
        return best_k, best_kmeans, cluster_label_k, score_list

class LevelCluster:
    def __init__(self):
        super(LevelCluster, self).__init__()
    def train(self, linkage_method:str, inp, n_cluster):
        first = linkage(inp, method=linkage_method)
        label_pred = fcluster(first, t=n_cluster, criterion='maxclust')
        return label_pred

class DBSCANCluster:
    def __init__(self):
        super(DBSCANCluster, self).__init__()
    def train(self, inp, eps:float, min_samples=100):
        cluster_label = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(inp)
        return cluster_label

class analyse_cluster:
    def __init__(self, log_dir):
        super(analyse_cluster, self).__init__()
        self.logger = load_logger(log_dir)

    def describe(self, inp, cluster_label, best_k):
        cluster_label = pd.DataFrame(cluster_label, columns=['cluster'])
        merge_data = pd.concat([inp, cluster_label], axis=1)
        ##聚类的样本量和占比
        clustering_count = pd.DataFrame(merge_data['color'].groupby(merge_data['cluster']).count()).T.rename({'color': 'count'})
        print(clustering_count)
        clustering_ratio = (clustering_count / len(merge_data)).round(2).rename({'count': 'percentage'})
        print(clustering_ratio)

        ##计算每个类别的特征
        cluster_feature = []
        cols = ['color', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 'language', 'country',
                'content_rating']
        cols_num = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes',
                    'actor_1_facebook_likes', 'gross', 'cast_total_facebook_likes',
                    'facenumber_in_poster', 'num_user_for_reviews', 'budget',
                    'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']
        for line in range(best_k):
            label_data = merge_data[merge_data['cluster'] == line]
            part1_data = label_data[cols_num]
            part1_desc = part1_data.describe().round(3)
            merge_data1 = part1_desc.iloc[2, :]

            part2_data = label_data[cols]
            part2_desc = part2_data.describe(include='all')
            merge_data2 = part2_desc.iloc[2, :]

            merge_line = pd.concat((merge_data1, merge_data2), axis=0)
            cluster_feature.append(merge_line)
        cluster_pd = pd.DataFrame(cluster_feature).T
        self.logger.info('{:*^60}'.format('每个类别主要特征：'))
        all_cluster_set = pd.concat((clustering_count, clustering_ratio, cluster_pd), axis=0)
        print(all_cluster_set)

        from sklearn.preprocessing import MinMaxScaler
        model_sacle = MinMaxScaler()
        num_sets = cluster_pd.iloc[:15, :].T.astype(np.float64)
        num_sets_max_min = model_sacle.fit_transform(num_sets)

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        labels = np.array(merge_data1.index)
        cor_list = ['g', 'r', 'y', 'b']
        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        for i in range(len(num_sets)):
            data_tmp = num_sets_max_min[i, :]
            data = np.concatenate((data_tmp, [data_tmp[0]]))
            ax.plot(angles, data, 'o-', c=cor_list[i%4], label="第%d类渠道"%(i))
            ax.fill(angles, data, alpha=2.5)
        ax.set_rlim(-0.2, 1,2)
        ax.set_title("各聚类类别显著特征对比")
        ax.set_thetagrids(angles*180/np.pi, labels, fontproperties="SimHei")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
        plt.show()
        plt.savefig('cluster.png')

        ## 多维数据可视化
        tsne_out = TSNE(n_components=2).fit_transform(merge_data[cols_num])
        pos = pd.DataFrame(tsne_out, columns=['X', 'Y'])
        ##数据可视化
        plt.figure()
        pos.plot(kind='scatter', x='X', y='Y', color='green')
        plt.show()
        ###聚类后的数据可视化
        pos['cluster'] = merge_data['cluster']
        plt.figure()
        cor_list = ['blue', 'green', 'black', 'red']
        ax = pos[pos['cluster'] == 0].plot(kind='scatter', x='X', y='Y', color='blue', label='第0类')

        for k in range(1, best_k):
            pos[pos['cluster'] == k].plot(kind='scatter', x='X', y='Y', color=cor_list[k%4], label='第%d类'%k, ax=ax)
        plt.show()

