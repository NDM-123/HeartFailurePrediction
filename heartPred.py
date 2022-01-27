import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score,confusion_matrix, recall_score, precision_score, classification_report
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import warnings
warnings.filterwarnings("ignore")

# class Model:
#     def __init__(self, model):
#         self.model = model
#         self.numFeatures = 0
#         self.accuracy = 0



def modelPrediction(model,X_train,X_test,y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    cv = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')

    test_acc = accuracy_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(type(model).__name__)
    print("cross validation: ", cv.mean())
    print("Train set Accuracy: ", train_acc)
    print("Test set Accuracy: ", test_acc)
    print()
    return test_acc

def data_process():
    df = pd.read_csv(r'heart.csv')
    df.head()

    # label encoding
    label_encoder = preprocessing.LabelEncoder()
    for i in df:
        df[i] = label_encoder.fit_transform(df[i])

    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    return x, y

if __name__ == '__main__':

        x, y = data_process()

        models = []

        X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2,random_state=0)

        # Before feature selection
        # DT = modelPrediction(DecisionTreeClassifier(max_depth=5),X_train, X_test, y_train, y_test)
        AB = modelPrediction(AdaBoostClassifier(),X_train, X_test, y_train, y_test)
        # AB = modelPrediction(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),X_train, X_test, y_train, y_test)
        # NN = modelPrediction(MLPClassifier(solver='adam'),X_train, X_test, y_train, y_test)
        # KNN = modelPrediction(KNeighborsClassifier(),X_train, X_test, y_train, y_test)
        # SVM = modelPrediction(SVC(),X_train, X_test, y_train, y_test)

        models.append(modelPrediction(DecisionTreeClassifier(max_depth=5),X_train, X_test, y_train, y_test))
        # models.append(modelPrediction(AdaBoostClassifier(),X_train, X_test, y_train, y_test))
        models.append(modelPrediction(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),X_train, X_test, y_train, y_test))
        models.append(modelPrediction(MLPClassifier(solver='adam'),X_train, X_test, y_train, y_test))
        models.append(modelPrediction(KNeighborsClassifier(),X_train, X_test, y_train, y_test))
        models.append(modelPrediction(SVC(),X_train, X_test, y_train, y_test))

        #   feature selection
        bestTestAccuracy = [0,0,0,0,0]
        bestNumberOfFeatures = [0,0,0,0,0]


        for k in range(1,x.shape[1]):
            answers = []
            skb = SelectKBest(chi2, k=k)
            skb.fit(x, y)
            X_new = skb.transform(x)
            X_train, X_test, y_train, y_test = train_test_split(X_new, y,test_size=0.2,random_state=0)


            print("After feature selection, num of features:" + str(k))
            # After feature selection
            answers.append(modelPrediction(DecisionTreeClassifier(max_depth=5),X_train, X_test, y_train, y_test))
            answers.append(modelPrediction(AdaBoostClassifier(DecisionTreeClassifier(max_depth=5)),X_train, X_test, y_train, y_test))
            answers.append(modelPrediction(MLPClassifier(hidden_layer_sizes=10, learning_rate_init=0.01, random_state=0,solver='adam'),X_train, X_test, y_train, y_test))
            answers.append(modelPrediction(KNeighborsClassifier(n_neighbors=8),X_train, X_test, y_train, y_test))
            answers.append(modelPrediction(SVC(),X_train, X_test, y_train, y_test))
            # answers.append(modelPrediction(SVC(C=10, gamma=0.5, kernel='poly', random_state=0, probability=True),X_train, X_test, y_train, y_test))

            for i in range(len(answers)):
                    if answers[i] > bestTestAccuracy[i]:
                        bestTestAccuracy[i] = answers[i]
                        bestNumberOfFeatures[i] = k
        # how come some features without feature selection get same score? KNN...
        print(models)
        print(bestTestAccuracy)
        print(bestNumberOfFeatures)