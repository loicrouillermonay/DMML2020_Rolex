# Data Mining and Machine Learning 2020 – Team Rolex

- Francis Ruckstuhl (16-821-738)
- Hanna Birbaum ()
- Loïc Rouiller-Monay (16-832-453)

## Video presentation

[![DM&ML2020 – Team Rolex](https://i.ytimg.com/vi/dBPvnDJUlF8/maxresdefault.jpg)](https://youtu.be/dBPvnDJUlF8 "DM&ML2020 – Team Rolex")


## Table of Contents
1. Introduction
2. Initial Setup 
3. Exploratory Data Analysis
4. Data Cleaning
5. Feature Engineering
6. Preprocessing
7. Models
8. Submissions

## 1. Introduction
### The premise of the project 
Real or Not? NLP with Disaster Tweets: Machine Learning model that can predict which tweets are about a real disaster and which are not. The project topic is based around a Kaggle competition.

### Goal
The goal of this group project is to build models that can predict whether or not a tweet it about a disaster. Our team aims to try out different models with a vast variety of different parameters in order to find the best model that obtains a high accuracy score.

### Our approach
For the final group project, we proceeded as follows: In order to exploit the full computational capacity available, we relied on our own computer performance by using Jupyter Notebook (this allowed us to build models that would crash on Google Colab). Once we all completed that initial setup, our team members split up the work to be done and created different branches on GitHub to work on their respective task. Being assigned with a specific type of algorithm, each member tried to build a model and maximize its accuracy score. During this process, the different parameters used to train the models and the respective results that were achieved were added to an Excel table. In addition, we organized several meetings per week in order to keep the others informed on our progress and to exchange our current results. Finally, the different branches were merged into one finalized notebook. 

## 2. Initial Setup
As a first step, the user is required to import the necessary libraries and packages as well as the datasets, which consist in the training and test set. Moreover, our team provides a set up to directly create submissions for the website AIcrowd and a brief description of the features in the present dataset.

## 3. Exploratory Data Analysis
As a next step, the data requires to be explored and analyzed. What is the distribution of the target value in the dataset at hand? Are there any missing values, and if so, where are they? How long are the tweets? Where to disasters tend to occur, where not? Our team provides additional plots that serve as visual tools in order to facilitate the process of EDA.

## 4. Data Cleaning
After the data has been explored, it is time to clean the data and to remove as much dirt and noise as possible. Our team uses the help of various tools to do so, such as Regular Expressions, Python functions, and imported packages.

## 5. Feature Engineering
Oftentimes, a dataset requires feature engineering alongside the steps of cleaning. Our team has attempted to further modify the values in the column “location” in the present dataset, as we noticed that the values contained a considerable amount of noise and inconsistencies throughout. Initially, we tried out different methods before stumbling upon a library called “pycountry” to create a little function. However, the aforementioned library was unable to correctly convert certain location names into their respective country. To name an example, it converted the ISO alpha code “UK”, into the country “Ukraine”, which is an incorrect conversion. Therefore, we eventually dropped “pycountry” as a tool to help us further clean the data. Instead, we decided to replace this step by the so-called process of "Data Augmentation". For that, we used the cleaned values from certain columns and merged them with the tweets in the colum "text" in order to feed our models with more information.

## 6. Preprocessing
Before the models can be built, the data requires the step of preprocessing. Here, the tweets are chopped into so-called tokens that will then be lowercase and lemmatized. Moreover, stopwords and additional tokens that are to be considered as noise are removed in this step.

## 7. Models
As a last step, the different models will be built. The Rolex team strived to try out different models and explore different parameters to find an optimal code that returns a high accuracy score.

### BOW
First, our team members build various BOW (“Bag of Words”) models using Logistic Regression and Decision Trees while varying with the independent variables in order to determine which features will increase the model’s accuracy.

### Word2Vec
Secondly, our team members build various Word2Vec models using Logistic Regression and, again, varying with the features being used to find the best accuracy for this model.

### TF-IDF
Lastly, the Rolex team built several TF-IDF models using Logistic Regression, Decision Trees, Random Forests and a kNN classifier. In order to boost the model’s accuracy, a Standard Scaler and Principal Component Analysis (“PCA”) were included in the process of construction.

## 8. Submissions
Finally, if the user wishes to do so, submission files can be created for the aforementioned models that were built in this notebook. These files can be submitted on AIcrowd.com

### Reported accuracies

<p align="center">
  <img src="https://github.com/loicrouillermonay/DMML2020_Rolex/blob/wrap-up/documents/plots/acc-plot.png" />
</p>

### Best submission

**Submission n° 10 : 0.8222%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 3), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

### Previous submissions

**Submission n° 1 : 0.808%**
- spacy_tokenizer v1*
- TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), tokenizer=spacy_tokenizer)
- LogisticRegression(solver='lbfgs', max_iter=1000)

**Submission n° 2 : 0.818%**
- feature engineering with num_chars, num_words, avg_words
- Data cleaning with unicode literals, urls, link, accounts and hashtags
- BOW
- LogisticRegression(solver='lbfgs', max_iter=1000)

**Submission n° 3 : 0.809%**
- feature engineering with num_chars, num_words, avg_words, num_hashtags
- Data cleaning with unicode literals, urls, link, author, hashtags, punctuations, lowercase
- BOW
- LogisticRegression(solver='lbfgs', max_iter=1000)

**Submission n° 4 : 0.801%**
- Data cleaning with unicode literals, urls, link, author, hashtags, punctuations, lowercase, lemmatize, stemming, stopwords
- Doc2Vec(dm=0, vector_size=30, negative=6, hs=0, min_count=1, sample=0, workers=cores, epoch=300)
- LogisticRegression(max_iter=1000, solver='lbfgs')

**Submission n° 5 : 0.812%**
- Data cleaning like submission n° 4 but without stemming 
- Doc2Vec(dm=0, vector_size=30, negative=6, hs=0, min_count=1, sample=0, workers=cores, epoch=300)
- LogisticRegression(max_iter=1000, solver='lbfgs')

**Submission n° 6 : 0.817%** 
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 2), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=5)

**Submission n° 7 : 0.82%** 
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 3), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 8 : 0.81%** 
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 5), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 9 : 0.818%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 2), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 10 : 0.8222%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 3), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 11 : 0.822%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 6), tokenizer=spacy_tokenizer)
- PCA(n_components=0.8)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 11 : 0.816%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 1), tokenizer=spacy_tokenizer)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 12 : 0.809%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 3), tokenizer=spacy_tokenizer)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

**Submission n° 13 : 0.806%** 
- feature engineering with concatenation of country, location, keyword and text
- spacy_tokenizer v2*
- TfidfVectorizer(ngram_range=(1, 3), min_df = 2, max_df = 0.99, tokenizer=spacy_tokenizer)
- LogisticRegressionCV(max_iter=5000, solver='lbfgs', cv=10)

##### *Spacy_Tokenizers versions
- *spacy_tokenizer v1 : lemmatize each token, convert each token into lowercase and remove stopwords*
- *spacy_tokenizer v2 : the one in Chapter 6 of the current jupyter notebook*
