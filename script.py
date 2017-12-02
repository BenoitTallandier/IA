from sklearn import tree
from sklearn.datasets import load_iris
import numpy
import csv


###############################
percTest = 0.3 # taille de l'echantillon de test
###############################


results = []
with open("dataB.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)


X = numpy.array(results)[1:,0:27]
Y = numpy.array(results)[1:,28]

total = X.shape[0];
train = int(total*(1-percTest))
X_train = X[0:train]
X_test = X[train:total]
y_train = Y[0:train]
y_test = Y[train:total]


clf_gini = tree.DecisionTreeClassifier(criterion = "entropy",splitter = "best", max_depth = None, min_samples_split = 2, min_samples_leaf = 1, min_weight_fraction_leaf = 0, max_features = None, random_state = None, max_leaf_nodes = 20, min_impurity_decrease = 0, class_weight = None, presort = False)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred_train = clf_gini.predict(X_train)

y_pred_train = numpy.asarray(map(int, y_pred_train))
y_train = numpy.asarray(map(int, y_train))

y_pred = numpy.asarray(map(int, y_pred))
y_test = numpy.asarray(map(int, y_test))


matriceConfusion = [0,0,0,0] 
# 0: vrai risque            1 : risque non detecte
#    
# 2: faux risque detecte    3 : pas de risque detecte (reel, predit)

for i in range(len(y_pred_train)):
    if(y_pred_train[i] == y_train[i] and y_train[i] == 0):
        matriceConfusion[3]+=1
    elif(y_pred_train[i] == y_train[i] and y_train[i] == 1):
        matriceConfusion[0]+=1
    elif(y_pred_train[i] != y_train[i] and y_train[i] == 0):
        matriceConfusion[2]+=1
    elif(y_pred_train[i] != y_train[i] and y_train[i] == 1):
        matriceConfusion[1]+=1
#matriceConfusion =  [i *100/len(y_test) for i in matriceConfusion]
print "train"
print(" %d       %d\n %d      %d"%(matriceConfusion[0],matriceConfusion[1],matriceConfusion[2],matriceConfusion[3]) )


matriceConfusion = [0,0,0,0] 

for i in range(len(y_pred)):
    if(y_pred[i] == y_test[i] and y_test[i] == 0):
        matriceConfusion[3]+=1
    elif(y_pred[i] == y_test[i] and y_test[i] == 1):
        matriceConfusion[0]+=1
    elif(y_pred[i] != y_test[i] and y_test[i] == 0):
        matriceConfusion[2]+=1
    elif(y_pred[i] != y_test[i] and y_test[i] == 1):
        matriceConfusion[1]+=1
#matriceConfusion =  [i *100/len(y_test) for i in matriceConfusion]
print "\ntest"
print(" %d       %d\n %d      %d"%(matriceConfusion[0],matriceConfusion[1],matriceConfusion[2],matriceConfusion[3]) )


tree.export_graphviz(clf_gini,out_file='tree.dot')
