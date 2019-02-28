---
title: "Topic Modelling using LDA"
excerpt: "In this noteboook we classifiy documents using a LDA model.<br/><img src='/images/500x300.png'>"
collection: portfolio
---


# Topic Modelling

Topic modelling is classification of documents into various topics.

In this notebook I have tried to classify a document into different topics using LDA(Latent Dirichilet Allocation). LDA is an unsupervied topic modelling algorithm.

I have initially trained a LDA model on 2000 Wikipedia articles dividing them into 15 topics.


```python
from gensim.models.ldamodel import LdaModel
from gensim.corpora import Dictionary
from util import WikipediaReader
from preprocess import remove_number,remove_punctuation,remove_stopword,multiple_whitespaces,preprocess_pipeline

lda = LdaModel.load('ldamodel')
dic = Dictionary.load('dictionary')
```

Printing 5 words from each topic to see the composition of each topic and decide a name for each topic


```python
print(lda.print_topics(num_topics=15, num_words=5))
```

    [(0, '0.016*"film" + 0.014*"music" + 0.009*"song" + 0.009*"album" + 0.007*"band"'), (1, '0.006*"christmas" + 0.006*"disease" + 0.005*"women" + 0.005*"health" + 0.005*"children"'), (2, '0.031*"german" + 0.024*"french" + 0.018*"la" + 0.015*"germany" + 0.014*"english"'), (3, '0.012*"war" + 0.010*"air" + 0.009*"army" + 0.009*"japanese" + 0.008*"aircraft"'), (4, '0.051*"displaystyle" + 0.042*"x" + 0.030*"r" + 0.029*"k" + 0.029*"c"'), (5, '0.008*"series" + 0.006*"character" + 0.005*"episode" + 0.004*"him" + 0.004*"characters"'), (6, '0.012*"king" + 0.009*"british" + 0.009*"john" + 0.008*"born" + 0.008*"church"'), (7, '0.018*"team" + 0.015*"season" + 0.015*"game" + 0.010*"league" + 0.009*"games"'), (8, '0.039*"school" + 0.029*"university" + 0.017*"students" + 0.017*"college" + 0.016*"education"'), (9, '0.011*"government" + 0.010*"party" + 0.008*"president" + 0.007*"political" + 0.006*"war"'), (10, '0.009*"isbn" + 0.008*"book" + 0.007*"century" + 0.006*"press" + 0.005*"published"'), (11, '0.016*"city" + 0.008*"south" + 0.008*"area" + 0.007*"north" + 0.007*"river"'), (12, '0.006*"power" + 0.005*"systems" + 0.005*"design" + 0.004*"energy" + 0.004*"model"'), (13, '0.007*"public" + 0.007*"government" + 0.006*"information" + 0.006*"services" + 0.005*"company"'), (14, '0.013*"species" + 0.009*"lord" + 0.008*"water" + 0.006*"planet" + 0.005*"small"')]


Seeing the words we can put the following names to each topic as.


|Topic Number|Description|
|------------|-----------|
|Topic 0|Movies|
|Topic 1|Health|
|Topic 2|Place|
|Topic 3|Military|
|Topic 4|Characters|
|Topic 5|TV Series|
|Topic 6|History and Religion|
|Topic 7|Sports|
|Topic 8|Education|
|Topic 9|Politics|
|Topic 10|Books & Literature|
|Topic 11|Location/City|
|Topic 12|Technology|
|Topic 13|Bussiness|
|Topic 14|Science|


```python
wikireader = WikipediaReader('data')
wikis,names = wikireader.read_all(True)
start=50
end=60
wiki = wikis[start:end]
name = names[start:end]
from os import listdir

topic_names= ['Movies','Health','Place','Military','Characters','TV Series','History & Religion','Sports','Education','Politics','Books&Literature','Location/City','Technology','Bussiness','Science']

pp=[remove_stopword,multiple_whitespaces,remove_number,remove_punctuation]
texts = [preprocess_pipeline(pp,x) for x in wiki]
texts = [x.split(" ") for x in texts]
wordidx = [dic.doc2bow(x) for x in texts]
#print(texts)
print(len(wordidx))

topics = lda.get_document_topics(wordidx)
print(topics)
for _,possible in enumerate(topics):
    print('---------------------')
    print(name[_])
    possible=list(possible)
    possible.sort(key=lambda x:x[1],reverse=True)
    print(possible)
    print(topic_names[possible[0][0]])
    print('---------------------')
```

    10
    <gensim.interfaces.TransformedCorpus object at 0x7f8c741bc160>
    ---------------------
    DavidDubinsky.txt
    [(9, 0.7594137), (13, 0.08005035), (5, 0.057598345), (11, 0.03786402), (8, 0.029469233), (6, 0.027811814)]
    Politics
    ---------------------
    ---------------------
    KetteringTownFC.txt
    [(7, 0.9542744), (11, 0.036167875)]
    Sports
    ---------------------
    ---------------------
    Bandrockandpop.txt
    [(0, 0.774972), (12, 0.14981812), (1, 0.05851231), (14, 0.016021645)]
    Movies
    ---------------------
    ---------------------
    Matchracing.txt
    [(7, 0.6470198), (3, 0.1519034), (12, 0.09600693), (14, 0.08260851), (11, 0.02164131)]
    Sports
    ---------------------
    ---------------------
    July1981.txt
    [(9, 0.2402989), (7, 0.16389076), (5, 0.12323976), (3, 0.1153526), (6, 0.114085205), (11, 0.0966594), (13, 0.042434987), (12, 0.036597762), (0, 0.026117383), (1, 0.016423088), (2, 0.014031778)]
    Politics
    ---------------------
    ---------------------
    KMPHTV.txt
    [(0, 0.4064288), (13, 0.2705764), (11, 0.26480734), (12, 0.05337839)]
    Movies
    ---------------------
    ---------------------
    ListoftributestoHankWilliams.txt
    [(0, 0.7830423), (5, 0.16522576), (6, 0.039263647), (11, 0.011793682)]
    Movies
    ---------------------
    ---------------------
    PrussianLithuanians.txt
    [(2, 0.4664168), (9, 0.19005), (10, 0.17846893), (6, 0.062037177), (11, 0.054187454), (1, 0.02081955), (14, 0.014304302), (8, 0.013308195)]
    Place
    ---------------------
    ---------------------
    StéphaneZagdanski.txt
    [(10, 0.59362227), (0, 0.14276719), (2, 0.118035965), (5, 0.115539074), (12, 0.02333492)]
    Books&Literature
    ---------------------
    ---------------------
    ListofInternetpioneers.txt
    [(13, 0.3599625), (8, 0.30366918), (12, 0.12216686), (0, 0.084459774), (6, 0.06493421), (10, 0.058096338)]
    Bussiness
    ---------------------

