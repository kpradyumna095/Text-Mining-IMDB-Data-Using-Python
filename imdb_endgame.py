# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 16:31:19 2019

@author: Hello
"""

from selenium import webdriver
browser = webdriver.Chrome()
from bs4 import BeautifulSoup as BS

##Movie name -- Avengers Endgame
page = "https://www.imdb.com/title/tt4154796/reviews?ref_=tt_ql_3"

##Importing few exceptions to surpass the error messages while extracting
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException

browser.get(page)
import time
reviews=[]

i=1
##The while loop which is written to load all the reviews required to the browser. We can do this,
##till the load button disappears
while(i>0):
    try:
# Storing the load more button page xpath which we will be using it for click it through selenium for loading few more review
        button = browser.find_element_by_xpath('//*[@id="load-more-trigger"]')
        button.click()
        time.sleep(5)
    except NoSuchElementException:
        break
    except ElementNotVisibleException:
        break
    
        
#Getting the page source for the entire imdb after loading all the reviews
ps = browser.page_source

#Converting page source and Beautiful soup object
soup = BS(ps,'html.parser')

#Extracting the reviews present in div html_tag having class containing "text" in its value
reviews = soup.findAll("div",attrs={"class","text"})
for i in range(len(reviews)):
    reviews[i] = reviews[i].text
    
##Creating a dataframe
import pandas as pd
import re
import nltk
movie_reviews = pd.DataFrame(columns = ["reviews"])
movie_reviews["reviews"]= reviews

movie_reviews.to_csv("endgame_reviews.csv",encoding = "utf-8")

reviews_endgame = ' '.join(reviews)

## Preforming sentence tokenization
from nltk.tokenize import sent_tokenize
token_sent = sent_tokenize(reviews_endgame)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
wordnet = WordNetLemmatizer()

filtered_split=[]
for i in range(len(token_sent)):
    review = re.sub("[^A-Za-z" "]+"," ",token_sent[i])
    review = re.sub("[0-9" "]+"," ",token_sent[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    filtered_split.append(review)
    
##Creating TFIDF model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
text_tf = tf.fit_transform(filtered_split)
feature_names = tf.get_feature_names()
dense = text_tf.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist,columns = feature_names)
df.iloc[:,0].name
df.drop(["_________________________________they"],axis=1,inplace=True)
df.iloc[:,0].name
df.drop(["_could_"],axis=1,inplace =True)


##plotting wordcloud on TFIDF
from wordcloud import WordCloud
import matplotlib.pyplot as plt

cloud = ','.join(df)

wordcloud =WordCloud(
        background_color='black',
        width = 1800,
        height = 1400).generate(cloud)
plt.imshow(wordcloud)

## importing positive words to plot positive word cloud

with open("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Text mining\\Datasets\\positive-words.txt","r") as pos:
    poswords = pos.read().split("\n")
    
poswords = poswords[36:]


posi_words =' '.join([w for w in df if w in poswords])

cloud_pos = WordCloud(
        background_color ='black',
        width = 1800,
        height =1400).generate(posi_words)
plt.imshow(cloud_pos)

##Importing negative words to plot negative word cloud

with open("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\Text mining\\Datasets\\negative-words.txt","r") as nos:
    negwords = nos.read().split("\n")
negwords = negwords[37:]

negi_words = ' '.join([w for w in df if w in negwords])

cloud_neg = WordCloud(
        background_color ='black',
        width =1800,
        height =1400).generate(negi_words)
plt.imshow(cloud_neg)
