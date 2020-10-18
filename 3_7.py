# -*- coding: utf-8 -*-
#导入文件
import pandas as pd
file1 = open('genome-scores.csv', encoding='utf-8')
file2 = open('genome-tags.csv', encoding='utf-8')
file3 = open('links.csv', encoding='utf-8')
file4 = open('movies.csv', encoding='utf-8')
file5 = open('ratings.csv', encoding='utf-8')
file6 = open('tags.csv', encoding='utf-8')

genome_scores = pd.read_csv(file1)
genome_tags   = pd.read_csv(file2)
links         = pd.read_csv(file3)
movies        = pd.read_csv(file4)
ratings       = pd.read_csv(file5)
tags          = pd.read_csv(file6)

#1
user1 = tags['userId']
user2 = ratings['userId']
user3 = user1.append(user2)
user  = user3.drop_duplicates()
print('一共有不同的用户数：',len(user))

#2
movie1 = genome_scores['movieId']
movie2 = links['movieId']
movie3 = movies['movieId']
movie4 = movie1.append(movie2)
movie5 = movie4.append(movie3)
movie  = movie5.drop_duplicates() 
print('一共有不同的电影数：',len(movie))

#3
genres0 = movies['genres']
genres1 = list(genres0.str.split('|'));
genres=list(set([i for j in genres1 for i in j ])) # 以空格为分隔符，分隔成两个
print('一共有不同的电影种类数量：',len(genres))

#4
tm = links['tmdbId']
tmna = tm.dropna()
d= len(links)-len(tmna)
print('没有外部链接的电影数',d)

#5
import time
time_start= time.strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
time_end  = time.strptime("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
timestamp_start = int(time.mktime(time_start))
timestamp_end   = int(time.mktime(time_end))
timestamp = ratings["timestamp"] 
ratings_2018 = ratings.loc[(timestamp < timestamp_end) & (timestamp>= timestamp_start)]
user_2018 = ratings_2018['userId']
user_2018 = user_2018.drop_duplicates()
print('2018年电影评分人数：',len(user_2018))

#6
mov = ratings_2018['movieId']
b = ratings_2018.groupby(mov)['rating'].mean().tolist()
mov_id = list(set(mov))
mov_id.sort()

from pandas.core.frame import DataFrame
new = {"movieId" : mov_id,
       "rating"  : b} 
data_ls=DataFrame(new)

time_start= time.strptime("2018-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
time_end  = time.strptime("2019-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
timestamp_start = int(time.mktime(time_start))
timestamp_end   = int(time.mktime(time_end))
timestamp = tags["timestamp"] 
tags_2018 = tags.loc[(timestamp < timestamp_end) & (timestamp>= timestamp_start)]
tagsf = tags_2018.groupby(['movieId'],as_index=False)['tag'].apply(lambda x:x.str.cat(sep=',')).reset_index()
tagsf.columns=['movieId','tags']
result0 = pd.merge(data_ls,tagsf,how = 'left',on='movieId')
result = result0[result0['rating']>=5]
result.to_csv('result.csv',encoding = 'utf-8-sig')
print('2018年评分5分以上的电影及其对应的标签已储存在表格result.csv')

#7 
ID = movies[movies['title']=="Avengers, The (2012)"]
aven = ratings_2018.loc[(ratings_2018['movieId'] == ID.iloc[0].at['movieId'])]
aven.index = pd.to_datetime(aven.index,unit='s')
data = aven.resample('M').mean()  #！！！

import matplotlib.pyplot as plt
x = data['rating']
plt.plot(x)


         
         
      