from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Input Data - [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# Output Data - male/female
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']
# Test Data
P = [[190, 70, 43], [150, 65, 35], [185, 75, 41]]

# Initialising all classifiers
classifiers = {
	'Decision Tree': DecisionTreeClassifier(),
	'Random Forest': RandomForestClassifier(),
	'K-Neighbors': KNeighborsClassifier(),
	'SVM': SVC(),
	'Perceptron Classifier': Perceptron(),
	'Logistic Regression': LogisticRegression(),
}

accuracy_list = []

for classifier_name, classifier in classifiers.items():
	print("----" + classifier_name + "----")
	clf = classifier
	# Fitting data in each of the classifier
	clf = clf.fit(X, Y)

	# Predicting for test data 
	prediction = clf.predict(P)
	total_prediction = clf.predict(X)

	# Calculating accuracy
	accuracy = accuracy_score(Y, total_prediction) 
	accuracy = accuracy * 100
	
	accuracy_list.append(accuracy)

	print("Prediction for", P[0], ":", prediction[0])
	print("Prediction for", P[1], ":", prediction[1])
	print("Prediction for", P[2], ":", prediction[2])
	print("Accuracy over whole dataset -", round(accuracy, 2), "%\n")

	# Storing accuracy for each classifier in the same dictionary
	classifiers[classifier_name] = accuracy

# Finding the maximum accuracy
max_accuracy = max(classifiers.values())
max_clf = [k for k, v in classifiers.items() if v == max_accuracy][0]

print("Highest accuracy of", max_accuracy, "% attained by", max_clf)