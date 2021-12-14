import pandas as pd
import numpy as np
from fastNLP import Instance, DataSet, Vocabulary
from fastNLP.io import Loader, Pipe, DataBundle
from sklearn.preprocessing import MinMaxScaler
class MVLoader(Loader):
    def __init__(self, demo=False):
        super(MVLoader, self).__init__()
        self.demo = demo

    def fillna_num(self, df: pd.DataFrame(), col_names: list):
        for col_name in col_names:
            df[col_name] = df[col_name].fillna(df[col_name].mean())
            # df[col_name] = df[col_name].fillna(df[col_name].median())
            ##插值法
            # df[col_name] = df[col_name].interpolate()
            df[col_name]= df[col_name].fillna(method='pad') ##前面只替换
            # df[col_name].fillna(method='backfill')
        return df
    def fillna_str(self, df: pd.DataFrame(), col_names: list):
        for col_name in col_names:
            #插值法
            df[col_name] = df[col_name].fillna(method='pad') ##前面只替换
            df[col_name] = df[col_name].fillna(method='backfill')
        return df

    def _load(self, path):
        ## 去除电影链接
        df = pd.read_csv(path).drop('movie_imdb_link', axis=1)
        df = self.fillna_num(df, ['num_critic_for_reviews', 'duration',
                                 'director_facebook_likes', 'actor_3_facebook_likes',
                                'actor_1_facebook_likes', 'gross',  'cast_total_facebook_likes',
                                  'facenumber_in_poster', 'num_user_for_reviews', 'budget',
                                  'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes'])
        df = self.fillna_str(df, ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name',
                                  'plot_keywords', 'language', 'country', 'content_rating', ])
        keys = list(df.keys())
        ds = DataSet()

        for _, row in df.iterrows():
            dict = {}
            content = ''
            for k, v in zip(keys, row):
                if k in dict.keys(): continue
                if k == 'imdb_score': dict[k] = v
                else: content += k+" is "+str(v)+" ; "
            dict['text'] = content
            if self.demo and len(ds) >= 30:
                break
            ds.append(Instance(**dict))
        return ds, df

    def load(self, paths) -> DataBundle:
        datasets = {}
        for name, path in paths.items():
            ds, _ = self._load(path)
            datasets[name] = ds
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle

class RegrePipe(Pipe):
    def __init__(self):
        super(RegrePipe, self).__init__()
    def process(self, data_bundle: DataBundle):
        train = data_bundle.get_dataset('train')
        dev = data_bundle.get_dataset('dev')
        test = data_bundle.get_dataset('test')
        data_bundle.apply_field(lambda x: x.split(' ')[:128], 'text', 'text')

        words = Vocabulary()
        words.from_dataset(train, field_name='text',  no_create_entry_dataset=[dev, test])
        words.index_dataset(train, dev, test, field_name='text', new_field_name='input_ids')
        data_bundle.apply_field(lambda x: float(x), field_name='imdb_score', new_field_name='imdb_score')
        data_bundle.apply_field(lambda x: x+(128-len(x))*[0], field_name='input_ids', new_field_name='input_ids')
        data_bundle.set_pad_val('input_ids', 0)
        data_bundle.set_input('input_ids')
        data_bundle.set_target('imdb_score')
        return data_bundle, words

    def process_from_file(self, path, demo=False) -> DataBundle:
        data_bundle = MVLoader(demo).load(path)
        return self.process(data_bundle)


class CorrelationPipe:
    def __init__(self):
        super(CorrelationPipe, self).__init__()

    def process(self, df: pd.DataFrame(), col_names: list):
        out = []
        for _, row in df.iterrows():
            for col_name in col_names:
                if col_name is ['genres', 'plot_keywords']:
                    out.extend(row[col_name].split('|'))
                else: out.append(row[col_name])
        return out

    def process_from_file(self, paths):
        pass

class ClusterPipe:
    def __init__(self):
        super(ClusterPipe, self).__init__()

    def process(self, df: pd.DataFrame()):
        cols = ['color', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 'language', 'country', 'content_rating']
        cols_num = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes',
                    'actor_1_facebook_likes', 'gross',  'cast_total_facebook_likes',
                    'facenumber_in_poster', 'num_user_for_reviews', 'budget',
                    'title_year', 'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio', 'movie_facebook_likes']
        df['genres'] = df['genres'].apply(lambda x: x.split('|')[0])
        # df['plot_keywords'] = df['plot_keywords'].apply(lambda x: x.split('|')[0])
        # for col in cols:
        #     words = df[col].unique()
        ##独热码
        ohmat = pd.get_dummies(df[cols])
        scale_mat = df[cols_num]
        scaler_model = MinMaxScaler()
        data_scale = scaler_model.fit_transform(scale_mat)
        out = np.hstack((data_scale, ohmat))
        return out


    def process_from_file(self, paths):
        pass

def split_mv(path, train_ratio=0.7, test_ratio=0.3):
    df = pd.read_csv(path)
    total_size = df.shape[0]
    train_size = int(train_ratio*total_size)
    dev_size = int(0.2*train_size)
    df_train = df[:train_size]
    df_test = df[train_size:]
    df_dev = df_train[-dev_size:]
    df_train = df_train[:-dev_size]
    df_train.to_csv(path.replace("movie_metadata.csv", "train.csv"), index=False, encoding="utf8")
    df_test.to_csv(path.replace("movie_metadata.csv", "test.csv"), index=False, encoding="utf8")
    df_dev.to_csv(path.replace("movie_metadata.csv", "dev.csv"), index=False, encoding="utf8")



if __name__ == '__main__':
    split_mv('../../../data/movie_metadata.csv')
    paths = {'train': '../../../data/train.csv', 'dev': '../../../data/dev.csv', 'test': '../../../data/test.csv'}
    ds = MVLoader().load(paths)
    # print(ds.get_dataset('train'))
    # ClusterPipe().process(df)
    RegrePipe().process(ds)