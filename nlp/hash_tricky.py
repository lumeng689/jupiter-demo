from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus=["I come to China to travel",
    "This is a car polupar in China",
    "I love tea and Apple ",
    "The work is to write some papers in science"]

vectorizer.fit(corpus)
print(vectorizer.fit_transform(corpus))
print(vectorizer.fit_transform(corpus).toarray())
print(vectorizer.get_feature_names())


from sklearn.feature_extraction.text import HashingVectorizer
vectorizer2=HashingVectorizer(n_features = 6,norm = None)
print(vectorizer2.fit_transform(corpus))
print(vectorizer2.fit_transform(corpus).toarray())
# print(vectorizer2.get_feature_names())