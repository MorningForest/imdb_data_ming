#-*- coding: utf-8 -*-
# Author: HW
# @Time: 2021/12/13 20:40

import pandas as pd
from sklearn.preprocessing import StandardScaler
from fastNLP import Instance, DataSet, Vocabulary

cols_num = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes',
            'actor_1_facebook_likes', 'gross', 'cast_total_facebook_likes',
            'facenumber_in_poster', 'num_user_for_reviews', 'budget',
            'title_year', 'actor_2_facebook_likes', 'aspect_ratio', 'movie_facebook_likes', 'imdb_score']

cols_str = ['color', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 'language', 'country',
            'content_rating', 'plot_keywords']

def fillna_num(df: pd.DataFrame(), col_names: list):
    for col_name in col_names:
        df[col_name] = df[col_name].fillna(df[col_name].mean())
        # df[col_name] = df[col_name].fillna(df[col_name].median())
        ##插值法
        # df[col_name] = df[col_name].interpolate()
        df[col_name] = df[col_name].fillna(method='pad')  ##前面只替换
        # df[col_name].fillna(method='backfill')
    return df


def fillna_str(df: pd.DataFrame(), col_names: list):
    for col_name in col_names:
        # 插值法
        df[col_name] = df[col_name].fillna(method='pad')  ##前面只替换
        df[col_name] = df[col_name].fillna(method='backfill')
    return df

def fn(x):
    if x > 7.2:  # 1218
        return 2
    elif x >= 6:  # 2371
        return 1
    else:
        return 0  # 1454

def fn1(ins):
    sent = """ the movie color is {} ; \
                actor_1_name is {} ; \
                actor_2_name is {} ; \
                actor_3_name is {} ; \
                language is {} ; \
                country is {} ; \
                content_rating is {} ; \
                genres is {} ; \
                plot_keywords is {} ;
           """.strip("\n").replace("    ", "")
    sent = sent.format(
        ins['color'],
        ins['actor_1_name'],
        ins['actor_2_name'],
        ins['actor_3_name'],
        ins['language'],
        ins['country'],
        ins['content_rating'],
        " , ".join(ins['genres'].split('|')),
        " , ".join(ins['plot_keywords'].split('|'))
    )
    return Instance(sent=sent)


def load_data(path):
    df = pd.read_csv(path)
    df_num = fillna_num(df[cols_num], cols_num)
    df_str = fillna_str(df[cols_str], cols_str)
    df_label = df_num['imdb_score']
    ssl = StandardScaler()
    df_matrix = ssl.fit_transform(df_num.drop(labels="imdb_score", axis=1))
    df_label = df_label.apply(fn)
    df_str = df_str.apply(fn1, axis=1)

    vocab = Vocabulary(padding=0)
    ds = DataSet(list(df_str))
    ds.apply_more(lambda x: {'token': x['sent'].split()})
    ds.add_field(fields=df_label, field_name="label")
    ds.add_field(fields=list(range(len(ds))), field_name="ins_id")
    ds_train, ds_test = ds.split(ratio=0.2, shuffle=False)
    vocab.from_dataset(ds_train, field_name='token', no_create_entry_dataset=ds_test)
    vocab.index_dataset(ds_train, ds_test, field_name='token', new_field_name="input_ids")
    ds_train.apply_field(lambda x: x + [0] * (128 - len(x)), field_name="input_ids", new_field_name="input_ids")
    ds_test.apply_field(lambda x: x + [0] * (128 - len(x)), field_name="input_ids", new_field_name="input_ids")
    ds_train.set_input("input_ids", "ins_id")
    ds_train.set_target("label")
    ds_test.set_input("input_ids", "ins_id")
    ds_test.set_target("label")
    return df_matrix, vocab, ds_train, ds_test
