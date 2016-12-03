# -*- coding: utf-8 -*-
"""
Created on Thu Sep 03 20:46:32 2015

@author: Cedric Oeldorf
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from __future__ import division
import time
import tweepy
from mpl_toolkits.basemap import Basemap
import re
from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import random



tweets_data_path = 'C:\Users\
Cedric Oeldorf\Desktop\UNIVERSITY\STK 353\New folder\news.txt'

with open('C:\\Users\\Cedric Oeldorf\\Desktop\\UNIVERSITY\
\STK 353\\New folder\\news.txt')as f:

    tweets_data = []
    for line in f:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except ValueError as e:
            print(e)
            pass
        
print "Number of tweets: " + str(len(tweets_data))

tweets = pd.DataFrame()
tweets['user'] = map(lambda tweet: \
tweet['user']['screen_name'] if 'user' in tweet else ' ', tweets_data):
    
tweets['user_followers_count'] = map(lambda tweet: \
tweet['user']['followers_count'] if 'user' in tweet else ' ', tweets_data)

tweets['text'] = map(lambda tweet: \
tweet['text'] if 'text' in tweet else ' ', tweets_data)

tweets['lang'] = map(lambda tweet: \
tweet['lang'] if 'lang' in tweet else ' ', tweets_data)

tweets['Location'] = [tweet['place']['country']if "place" in tweet \
 and tweet['place'] else np.nan for tweet in tweets_data]
 
tweets['time_zone'] = [tweet["user"]['time_zone'] if "user" in tweet \
and tweet["user"]['time_zone'] else np.nan for tweet in tweets_data]

tweets['retweet_count'] = [tweet["retweeted_status"]['retweet_count'] \
if "retweeted_status" in tweet and tweet["retweeted_status"]['retweet_count']
                       else '0' for tweet in tweets_data]   
                       
tweets['favorite_count'] = [tweet["retweeted_status"]['favorite_count'] \
if "retweeted_status" in tweet and tweet["retweeted_status"]['favorite_count']
                       else '0' for tweet in tweets_data]    
                       
tweets['created_at'] = map(lambda tweet: \
time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(tweet['created_at'],\
'%a %b %d %H:%M:%S +0000 %Y')) if 'created_at' in tweet else ' ', tweets_data)


#------------------------------------------------------------------------
#Processing and plotting of language data

tweets_by_lang = pd.DataFrame(tweets['lang'].value_counts())
tweets_by_lang['lang'] = tweets_by_lang.index
tweets_by_lang['count'] = tweets['lang'].value_counts()
tweetlangplot = tweets_by_lang[:15]

def label_lang (row):
    if row['lang'] == "en" :
        return 'English'
    if row['lang'] == "es" :
        return 'Spanish'        
    if row['lang'] == "und" :
        return 'Undefined' 
    if row['lang'] == "ar" :
        return 'Arabic'
    if row['lang'] == "tr" :
        return 'Turkish'        
    if row['lang'] == "in" :
        return 'Indonesian'        
    if row['lang'] == "pt" :
        return 'Portuguese'        
    if row['lang'] == "ja" :
        return 'Japanese'        
    if row['lang'] == "ru" :
        return 'Russian'        
    if row['lang'] == "ur" :
        return 'Urdu'
    if row['lang'] == "de" :
        return 'German'
    if row['lang'] == "hi" :
        return 'Hindi'
    if row['lang'] == "fr" :
        return 'French'
    if row['lang'] == "ne" :
        return 'Nepali'
    if row['lang'] == "th" :
        return 'Thai'
    if row['lang'] == "cy" :
        return 'Welsh'
    if row['lang'] == "fa" :
        return 'Farsi'
    if row['lang'] == "tl" :
        return 'Tagalog'
    if row['lang'] == "ko" :
        return 'Korean'   
            
tweetlangplot.apply (lambda row: label_lang (row),axis=1)        
tweetlangplot['Language'] = \
tweetlangplot.apply (lambda row: label_lang (row),axis=1)         

rel = [element for element in (tweetlangplot['count'].values)]
theSum = sum(rel)
rel2 = []
for x in rel:
  rel2.append(float(x/theSum))
tweetlangplot['Relative Frequency'] = rel2

sns.set_style("whitegrid")
bar_plot = sns.barplot(x=tweetlangplot["Language"],y= \
tweetlangplot["Relative Frequency"],
                        palette="muted",
                        x_order=tweetlangplot["Language"].tolist())
bar_plot.set(xlabel="News Organizations")
plt.title("Relative Frequency Distribution Language")                          
plt.xticks(rotation=40, size=9)
plt.show()

#------------------------------------------------------------------------
#processing and plotting of geographic data

tweets_by_location['Location'] = \
pd.DataFrame(tweets['Location'].value_counts())


m = Basemap(projection='robin',lon_0=0,resolution='c')
m.drawmapboundary(fill_color='#85A6D9')
#m.fillcontinents(color='white',lake_color='#85A6D9')
#m.drawcoastlines(color='#6D5F47', linewidth=.4)
#m.drawcountries(color='#6D5F47', linewidth=.4)
m.bluemarble()
lats = [54.0,38.0,52.5,39,-10,20,40,53,60,10,23,8,52,8,51,-30,46,\
25,-27,24,36,1,-2,15,60,-10,42.8333,-5,-41,-29]
lngs = [-2.0,-97.0,5.75,35,-55,77,-4,-8,-95,8,-102,-66,20,-2,9,-71,\
2,45,133,54,138,38,-77.5,-86.5,100,-76,12.8333,120,174,24]
populations = [292,246,20,20,17,16,16,15,14,11,10,10,10,7,7,7,6,5,5,5,\
5,5,4,4,3,3,3,3,2,2] 

x,y = m(lngs,lats)
s_populations = [p * p for p in populations]
m.scatter(
    x,
    y,
    s=s_populations,
    c='white', 
    marker='o', 
    alpha=0.25, 
    zorder = 2, 
    )
for population, xpt, ypt in zip(populations, x, y):
    label_txt = int(round(population, 0))
    plt.text(
        xpt,
        ypt,
        label_txt,
        color = 'white',
        size='small',
        horizontalalignment='center',
        verticalalignment='center',
        zorder = 3,
        )
plt.title('Top Locations Tweeted From')
plt.show()
plt.savefig("world.png", dpi=1000)

#------------------------------------------------------------------------
#split original tweets by retweets + plot

original = [element for element in tweets['text'].values if \
not element.startswith('RT')]

print "Number of Original : " + str(len(original))

retweet = [element for element in tweets['text'].values if \
element.startswith('RT')]
print "Number of Retweets : " + str(len(retweet))

rt = pd.DataFrame()
rt['number'] = (39531,34404)
rt['vs']= ('Original','Retweet')

sns.set_style("whitegrid")
bar_plot = sns.barplot(x=rt['vs'],y=rt['number'],
                        palette="muted",
                        x_order=tweetlangplot["Language"].tolist())
plt.title("Original vs Retweets")                          
plt.xticks(rotation=45)
plt.show()

#------------------------------------------------------------------------
#Distribution of news stations

def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)
    if match:
        return True
    return False
    
tweets['CNN'] = tweets['text'].apply(lambda tweet: word_in_text('cnn', tweet))
tweets['BBC'] = tweets['text'].apply(lambda tweet: word_in_text('bbc', tweet))
tweets['SkyNews'] = tweets['text'].apply(lambda tweet: \
word_in_text('skynews', tweet))
tweets['AlJazeera'] = tweets['text'].apply(lambda tweet: \
word_in_text('aljazeera', tweet))


print tweets['CNN'].value_counts()[True]
print tweets['BBC'].value_counts()[True]
print tweets['SkyNews'].value_counts()[True]
print tweets['AlJazeera'].value_counts()[True]
tweets_stations = \
[tweets['CNN'].value_counts()[True], tweets['BBC'].value_counts()[True], \
tweets['SkyNews'].value_counts()[True], \
tweets['AlJazeera'].value_counts()[True]]
corp = ['CNN','BBC','SkyNews','AlJazeera']

sumnews = sum(tweets_stations)
newsrelative = []
for x in tweets_stations:
  newsrelative.append(float(x/sumnews))

sns.set_style("whitegrid")
bar_plot = sns.barplot(x=corp,y=newsrelative,
                        palette="muted")  
bar_plot.set(xlabel="News Organizations")
plt.title("Relative Frequency Distribution News Organizations (Raw Data)")                          
plt.xticks(rotation=45)
sns.despine(left=True, bottom=True)
plt.show()


#------------------------------------------------------------------------
#Plotting distribution of news station using relevant tweets


tweets['refugees'] = \
tweets['text'].apply(lambda tweet: word_in_text('refugees', tweet))
tweets['migrants'] = \
tweets['text'].apply(lambda tweet: word_in_text('migrants', tweet))
tweets['syria'] = \
tweets['text'].apply(lambda tweet: word_in_text('syria', tweet))
tweets['syrian'] = \
tweets['text'].apply(lambda tweet: word_in_text('syrian', tweet))
tweets['refugee'] = \
tweets['text'].apply(lambda tweet: word_in_text('refugee', tweet))
tweets['crisis'] = \
tweets['text'].apply(lambda tweet: word_in_text('crisis', tweet))

tweets['relevant'] = tweets['text'].apply(lambda tweet: word_in_text('refugees', tweet) 
or word_in_text('migrants', tweet) or word_in_text('syria', tweet)
or word_in_text('syrian', tweet) or word_in_text('refugee', tweet) 
or word_in_text('crisis', tweet))

print tweets['refugees'].value_counts()[True]
print tweets['migrants'].value_counts()[True]
print tweets['syria'].value_counts()[True]
print tweets['syrian'].value_counts()[True]
print tweets['refugee'].value_counts()[True]
print tweets['crisis'].value_counts()[True]
print tweets['relevant'].value_counts()[True]


print tweets[tweets['relevant'] == True]['CNN'].value_counts()[True]
print tweets[tweets['relevant'] == True]['BBC'].value_counts()[True]
print tweets[tweets['relevant'] == True]['SkyNews'].value_counts()[True]
print tweets[tweets['relevant'] == True]['AlJazeera'].value_counts()[True]

relevantnews = [tweets[tweets['relevant'] == True]['CNN'].value_counts()[True],
tweets[tweets['relevant'] == True]['BBC'].value_counts()[True],
tweets[tweets['relevant'] == True]['SkyNews'].value_counts()[True],
tweets[tweets['relevant'] == True]['AlJazeera'].value_counts()[True]]

sumrelevant = sum(relevantnews)
relativerelevant = []
for x in relevantnews:
  relativerelevant.append(float(x/sumrelevant))

sns.set_style("whitegrid")
bar_plot = sns.barplot(x=corp,y=relativerelevant,
                        palette="muted")
bar_plot.set(xlabel="News Organizations")
plt.title("Relative Frequency Distribution of Refugee Crisis Coverage")                          
plt.xticks(rotation=45)
sns.despine(left=True, bottom=True)
plt.show()

#-----------------------------------------------------------------
#Plotting distribution of news stations using spanish tweets only


tweets['refugiado'] = \
tweets['text'].apply(lambda tweet: word_in_text('refugiado', tweet))
tweets['migrante'] = \
tweets['text'].apply(lambda tweet: word_in_text('migrante', tweet))
tweets['Siria'] = \
tweets['text'].apply(lambda tweet: word_in_text('siria', tweet))
tweets['sirios'] = \
tweets['text'].apply(lambda tweet: word_in_text('sirios', tweet))
tweets['integrar'] = \
tweets['text'].apply(lambda tweet: word_in_text('integrar', tweet))
#tweets['arabic'] = tweets[tweets['lang'] == 'ar']['text']

tweets['spanish_arabic_relevant'] = \
tweets['text'].apply(lambda tweet: word_in_text('refugiado', tweet) 
or word_in_text('inmigrante', tweet) or word_in_text('siria', tweet) 
or word_in_text('sirios', tweet) or word_in_text('integrar', tweet))

print tweets['refugiado'].value_counts()[True]
print tweets['migrante'].value_counts()[True]
print tweets['Siria'].value_counts()[True]
print tweets['sirios'].value_counts()[True]
print tweets['integrar'].value_counts()[True]
print tweets['spanish_arabic_relevant'].value_counts()[True]


print \
tweets[tweets['spanish_arabic_relevant'] == True]['CNN'].value_counts()[True]
print \
tweets[tweets['spanish_arabic_relevant'] == True]['BBC'].value_counts()[True]
print \
tweets[tweets['spanish_arabic_relevant'] == True]['SkyNews'].value_counts()[True]
print \
tweets[tweets['spanish_arabic_relevant'] == \
 True]['AlJazeera'].value_counts()[True]

spanishnews = \
[tweets[tweets['spanish_arabic_relevant'] == True]['CNN'].value_counts()[True],
tweets[tweets['spanish_arabic_relevant'] == True]['BBC'].value_counts()[True],
0,
tweets[tweets['spanish_arabic_relevant'] == \
True]['AlJazeera'].value_counts()[True]]

sumspanish = sum(spanishnews)
spanish = []
for x in spanishnews:
  spanish.append(float(x/sumspanish))

sns.set_style("darkgrid")
bar_plot = sns.barplot(x=corp,y=spanish,
                        palette="muted")                        
bar_plot.set(xlabel="News Organizations")
plt.title("Relative Frequency Distribution of Spanish")  
plt.xticks(rotation=45)
sns.despine(left=True, bottom=True)
plt.show()


#------------------------------------------------------------------------
#plotting news distribution using arabic tweets


tweets['arabic'] = tweets[tweets['lang'] == 'ar']['text']
arab = pd.DataFrame()
arab['arabic'] = tweets[tweets['lang'] == 'ar']['text']
arab['CNN'] = arab['arabic'].apply(lambda tweet: word_in_text('cnn', tweet))
arab['BBC'] = arab['arabic'].apply(lambda tweet: word_in_text('bbc', tweet))
arab['SkyNews'] = \
arab['arabic'].apply(lambda tweet: word_in_text('skynews', tweet))
arab['AlJazeera'] = \
arab['arabic'].apply(lambda tweet: word_in_text('aljazeera', tweet))

arabic= [arab['CNN'].value_counts()[True],
arab['BBC'].value_counts()[True],
arab['SkyNews'].value_counts()[True],
arab['AlJazeera'].value_counts()[True]]

sumarabic = sum(arabic)
arabics = []
for x in arabic:
  arabics.append(float(x/sumarabic))
  
sns.set_style("darkgrid")
bar_plot = sns.barplot(x=corp,y=arabics,
                        palette="muted")  
bar_plot.set(xlabel="News Organizations")
plt.title("Relative Frequency Distribution of Arabic")                        
plt.xticks(rotation=90)
sns.despine(left=True, bottom=True)
plt.show()


#------------------------------------------------------------------------
#Extracting and plotting most influential users

followers = pd.DataFrame(tweets['user_followers_count'])
followers['user'] = tweets['user']
followers = followers.sort('user_followers_count', ascending=False) 

followers = \
followers.groupby('user', group_keys=False).apply(lambda x: \
x.ix[x.user_followers_count.idxmax()])

followers = followers.sort('user_followers_count', ascending=False) 
follow['count'] = follow['user_followers_count']
follow = followers[:10]

sns.set_style("darkgrid")
bar_plot = sns.barplot(x=follow['user'],y=follow['user_followers_count'],
                        palette="muted")  
bar_plot.set(xlabel="Username")
plt.title("Most Influencial Users by Number of Followers")                        
plt.xticks(rotation=90)
sns.despine(left=True, bottom=True)
plt.show()


#------------------------------------------------------------------------
#word cloud

english = pd.DataFrame(tweets[tweets['lang'] == 'en']['text'])
englist = []
englist = english['text'].tolist()

def recursive_ascii_encode(lst):
    ret = []
    for x in lst:
        if isinstance(x, basestring):  # covers both str and unicode
            ret.append(x.encode('ascii', 'ignore'))
        else:
            ret.append(recursive_ascii_encode(x))
    return ret
englist = recursive_ascii_encode(englist)

text = " ".join(englist)

no_urls_no_tags = " ".join([word for word in text.split()
                                if 'http' not in word
                                    and not word.startswith('@')
                                    and word != 'RT'
                                ])
moz_mask = imread("./twitter_mask.png", flatten=True)


wc = WordCloud(background_color="white", font_path= \
"C:\\Windows.old\\WINDOWS\\Fonts\\verdana.ttf", stopwords= \
STOPWORDS, width=1800,
               height=140, mask=moz_mask)
wc.generate(no_urls_no_tags)
plt.imshow(wc)
plt.axis("off")
plt.savefig('mozsprint.png', dpi=300)


#------------------------------------------------------------------------
#tf-idf coding excluding tfidf.py functions because that script was given

import json
import pandas as pd
import tfidf
import numpy as np
import re
#text file containing tweets
tweets_data_path = 'allnews.txt'
tweets_data = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        tweet = json.loads(line)
        tweets_data.append(tweet)
    except:
        continue
    
tweets = pd.DataFrame()
tweets['text'] = \
map(lambda tweet:tweet['text'] if 'text' in tweet else ' ', tweets_data)
tweets['lang'] = \
map(lambda tweet:tweet['lang'] if 'lang' in tweet else ' ', tweets_data)

english = pd.DataFrame(tweets[tweets['lang'] == 'en']['text'])
#write tweets to a list
lst_tweets = list(english['text'])

#create instance of a table
table = tfidf.tfidf()

cnt = 0
for t in lst_tweets:
    #remove special characters in tweet
    t = re.sub(r'[^a-zA-Z0-9\n\.]', ' ', t)
    table.addDocument(cnt, t.split())
    cnt = cnt + 1

#Query words:
query_words = ['cnn', 'bbc', 'skynews','aljazeera']    
#query_words = ['refugee', 'migrant', 'syria']  

#write tf-idf table to a list
df = pd.DataFrame( table.similarities(query_words))
df.columns = ['tweet index', 'tf-idf']
#sort 
sorted_df = df.sort_index(by=['tf-idf'], ascending=[False])
top_tweets = [lst_tweets[a] for a in list(sorted_df['tweet index'].iloc[0:10])]
for t in top_tweets:
    print t

top_tfidf = list(sorted_df['tf-idf'].iloc[0:10])