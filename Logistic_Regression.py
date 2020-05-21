"""
File Name: Logistic_Regression.py
    The program uses Logistic Regression approach to predict whether a comment is toxic.
"""
import pandas
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_union

# All the relevant files
file_names = ["toxicity_annotated_comments.tsv", "toxicity_annotations.tsv"]

# Reads both the files and stores into pandas Dataframe
toxicity_annotations = pandas.read_csv(file_names[1], sep= '\t')
toxicity_annotated_comments = pandas.read_csv(file_names[0], sep= '\t')

# Takes the mean of the toxicity_score and gives a label True if greater than 0.5 else False
labels = toxicity_annotations.groupby('rev_id')['toxicity_score'].mean() > 0.5

# Adds the new column containing real comments with the above labels value
toxicity_annotated_comments['is_toxic'] = list(labels)

X_train = []    # Stores input training samples
X_test = []     # Stores input testing samples
Y_train = []    # Stores output samples for training data
Y_test = []     # Stores output samples for testing data

# The loop splits the data into train and test
for index, row in toxicity_annotated_comments.iterrows():
    if (row['split'] == "train"):
        X_train.append(row['comment'])
        Y_train.append(row['is_toxic'])

    elif (row['split'] == "test"):
        X_test.append(row['comment'])
        Y_test.append(row['is_toxic'])

# TfidfVectorizer creates word vectors for keeping track of word frequencies
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), max_features=10000, sublinear_tf=True)
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 4),
    max_features=30000)

# Combines the two vectorizers
vectorizer = make_union(tfidf, char_vectorizer)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

scores = []
labels = ['0', '1']
for label in labels:
    # build classifier
    LR = LogisticRegression(solver='saga')

    # predict the labels
    LR.fit(X_train, Y_train)

predictions = LR.predict(X_test)
print('Accuracy score: ', accuracy_score(Y_test, predictions))
