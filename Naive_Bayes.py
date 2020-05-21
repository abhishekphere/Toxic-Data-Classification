"""
File Name: Naive_Bayes.py
    The program uses Multinomial Naive Bayes approach to predict whether a comment is toxic.
"""
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WhitespaceTokenizer

# All the relevant files
file_names = ["toxicity_annotated_comments.tsv", "toxicity_annotations.tsv"]

# Reads both the files and stores into pandas Dataframe
toxicity_annotations = pandas.read_csv(file_names[1], sep= '\t')
toxicity_annotated_comments = pandas.read_csv(file_names[0], sep= '\t')

# remove rows with missing values
toxicity_annotations.dropna()
toxicity_annotated_comments.dropna()

# Takes the mean of the toxicity_score and gives a label True if greater than 0.5 else False
labels = toxicity_annotations.groupby('rev_id')['toxicity_score'].mean() > 0.5

# Adds the new column containing real comments with the above labels value
toxicity_annotated_comments['is_toxic'] = list(labels)

X_train = []    # Stores input training samples
X_test = []     # Stores input testing samples
Y_train = []    # Stores output samples for training data
Y_test = []     # Stores output samples for testing data

w_tokenizer = WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    """
    This function is responsible for lemmatization of text.
    :param text: Line of text
    :return: lemmetized string
    """
    text = [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
    return ' '.join(text)

toxicity_annotated_comments['comment'] = toxicity_annotated_comments.comment.apply(lemmatize_text)

# The loop splits the data into train and test
for index, row in toxicity_annotated_comments.iterrows():
    if (row['split'] == "train"):

        X_train.append(row['comment'])
        Y_train.append(row['is_toxic'])

    elif (row['split'] == "test"):

        X_test.append(row['comment'])
        Y_test.append(row['is_toxic'])

# Converts each comment into word count vectors
cv = CountVectorizer(lowercase=True, stop_words='english')

# Converts to the desired input format
X_train_cv = cv.fit_transform(X_train)

# Converts to the desired output format
X_test_cv = cv.transform(X_test)

# Creates the Multimonial Naive Bayes model and gives predictions for the test set
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, Y_train)
predictions = naive_bayes.predict(X_test_cv)

print('Naive Bayes Accuracy score: ', accuracy_score(Y_test, predictions))
print('Naive Bayes Recall score: ', recall_score(Y_test, predictions))
print('Naive Bayes F measure: ', f1_score(Y_test, predictions))
print('Naive Bayes precision score: ', precision_score(Y_test, predictions))
print()

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_cv, Y_train)
predictions = decision_tree.predict(X_test_cv)
print('Decision Tree Accuracy score: ', accuracy_score(Y_test, predictions))
print()

# Adaboost Tree Classifier
adaboost = AdaBoostClassifier()
adaboost.fit(X_train_cv, Y_train)
predictions = adaboost.predict(X_test_cv)
print('Adaboost Accuracy score: ', accuracy_score(Y_test, predictions))
print()

# K Nearest Neighbor classfier
k_nearest = KNeighborsClassifier()
k_nearest.fit(X_train_cv, Y_train)
predictions = k_nearest.predict(X_test_cv)
print('K-Nearest Neighbour Accuracy score: ', accuracy_score(Y_test, predictions))
print()

# Random Forest classifier
random_forest = RandomForestClassifier()
random_forest.fit(X_train_cv, Y_train)
predictions = random_forest.predict(X_test_cv)
print('Random Forest Accuracy score: ', accuracy_score(Y_test, predictions))
print()