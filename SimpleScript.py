from sklearn import tree
from sklearn.datasets import load_iris
import numpy
import csv
import math
from subprocess import check_call

###############################
percTest = 0.2   # taille de l'echantillon de test
hauteur = 30
max_leaf_number = 100;
###############################


def calcule_matrice_confusion(y_pred,y_reel):
    matriceConfusion = [0,0,0,0]
    for i in range(len(y_pred)):
        if(y_pred[i] == y_reel[i] and y_reel[i] == 0):
            matriceConfusion[3]+=1
        elif(y_pred[i] == y_reel[i] and y_reel[i] == 1):
            matriceConfusion[0]+=1
        elif(y_pred[i] != y_reel[i] and y_reel[i] == 0):
            matriceConfusion[2]+=1
        elif(y_pred[i] != y_reel[i] and y_reel[i] == 1):
            matriceConfusion[1]+=1
    #matriceConfusion =  [i *100/len(y_test) for i in matriceConfusion]
    print(" %d       %d\n %d      %d"%(matriceConfusion[0],matriceConfusion[1],matriceConfusion[2],matriceConfusion[3]) )
    return matriceConfusion


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

y_train = numpy.asarray(map(int, y_train))


clf_gini = tree.DecisionTreeClassifier(criterion = "gini",splitter = "best", max_depth = 100, min_samples_split = 15, min_samples_leaf = 5, min_weight_fraction_leaf = 0, max_features = None, random_state = None, max_leaf_nodes = 200, min_impurity_decrease = 0, class_weight = {0:2,1:98}, presort = False, min_impurity_split = 0)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_pred = numpy.asarray(map(int, y_pred))

matriceConfusion = calcule_matrice_confusion(y_pred,y_test)


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



def calcul_precision_rappel(matriceConfusion):
    #calcul precision positif
    if (matriceConfusion[0] + matriceConfusion[1] != 0):
        precision = float( matriceConfusion[0]) / ( matriceConfusion[0] + matriceConfusion[1])
    else:
        precision = 0
    print ("Rappel des positifs : %f" %(precision))
    if (matriceConfusion[0] + matriceConfusion[2] != 0):
        rappel = float( matriceConfusion[0]) / ( matriceConfusion[0] + matriceConfusion[2])
    else:
        rappel = 0
    print ("Precision des positifs : %f" %rappel)
    #calcule precision negatif
    if (matriceConfusion[3] + matriceConfusion[2] != 0):
        precision = float( matriceConfusion[3]) / ( matriceConfusion[3] + matriceConfusion[2])
    else:
        precision = 0
    print ("Rappel des negatif : %f " %(precision))
    if (matriceConfusion[3] + matriceConfusion[1] != 0):
        rappel = float( matriceConfusion[3]) / ( matriceConfusion[3] + matriceConfusion[1])
    else :
        rappel = 0
    print ("Precision des negatif : %f" %(rappel))


print ("\nentrainement")
matriceConfusion = calcule_matrice_confusion(y_pred_train,y_train);
calcul_precision_rappel(matriceConfusion);

print (" \n======================================== \n")
print ("\ntest")
matriceConfusion = calcule_matrice_confusion(y_pred,y_test);
calcul_precision_rappel(matriceConfusion);
tree.export_graphviz(clf_gini,out_file='tree.dot')
check_call(['dot','-Tpng','tree.dot','-o','tree.png'])