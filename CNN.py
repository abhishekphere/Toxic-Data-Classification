"""
File Name: CNN.py
    The program uses CNN approach to predict whether a comment is toxic.
"""
import pandas
from sklearn.feature_extraction.text import CountVectorizer

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

# Converts each comment into word count vectors
cv = CountVectorizer(lowercase=True, stop_words='english')
# Converts to the desired input format
X_train_cv = cv.fit_transform(X_train)

# Converts to the desired output format
X_test_cv = cv.transform(X_test)


# MLP classifier of sklearn
from sklearn.neural_network import MLPClassifier

#Configuration of CNN classifier
mlp = MLPClassifier(hidden_layer_sizes=(2,3,5), max_iter=50, alpha=1e-4,
                    solver='sgd', verbose=True, tol=0.00000001,
                    learning_rate_init=0.1)

mlp.fit(X_train_cv, Y_train)
print("Training set score: %f" % mlp.score(X_train_cv, Y_train))
print("Test set score: %f" % mlp.score(X_test_cv, Y_test))

#Evaluation metric of classifier
from sklearn.metrics import classification_report, confusion_matrix
predictions = mlp.predict(X_test_cv)
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))

