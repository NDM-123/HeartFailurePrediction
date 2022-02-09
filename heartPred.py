import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from IPython.display import display as disDF
import seaborn as sns
import matplotlib. pyplot as plt
from sklearn.metrics import RocCurveDisplay
import warnings
warnings.filterwarnings("ignore")


def plotFeatures(df,string):
    sns.pairplot(df, hue="HeartDisease", diag_kws={'bw': 0.2}, plot_kws={"s": 40}, corner=True)
    name = string + " normalizing.png"
    plt.savefig(name)

    # plt.show()

def modelPrediction(model,X_train,X_test,y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    cv = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    test_acc = accuracy_score(y_test, y_test_pred)
    train_acc = accuracy_score(y_train, y_train_pred)

    # print(type(model).__name__)
    # print("cross validation: ", cv.mean())
    # print("Train set Accuracy: ", train_acc)
    # print("Test set Accuracy: ", test_acc)
    # print()
    return test_acc

def data_process():
    df = pd.read_csv(r'heart.csv')

    # optional
    plotFeatures(df,'before')

    # label encoding + normalizing the data, not the best method as its better to use hot encode rather than label encoder
    label_encoder = preprocessing.LabelEncoder()
    for i in df:
        df[i] = label_encoder.fit_transform(df[i])

    plotFeatures(df,'after')

    print(df.describe().T)

    x = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    return x, y

def models(X_train, X_test, y_train, y_test):
    models = []

    models.append(modelPrediction(DecisionTreeClassifier(max_depth=6, random_state = 42), X_train, X_test, y_train, y_test))
    # n_estimators=1 == Decision tree
    models.append(modelPrediction(AdaBoostClassifier(n_estimators=5, base_estimator=DecisionTreeClassifier(max_depth=6), learning_rate=0.001,random_state = 42),X_train, X_test, y_train, y_test))
    models.append(modelPrediction(MLPClassifier(hidden_layer_sizes=32, learning_rate_init=0.01, random_state=0, solver='adam'), X_train, X_test, y_train, y_test))
    models.append(modelPrediction(KNeighborsClassifier(n_neighbors=8, algorithm='brute'), X_train, X_test, y_train, y_test))
    models.append(modelPrediction(SVC(), X_train, X_test, y_train, y_test))
    # print("GNB: ")
    # print(modelPrediction(GaussianNB(),X_train, X_test, y_train, y_test))
    return models

def modelsFeatureSelection(x,y,testPercent):
    #   feature selection
    bestTestAccuracy = [0,0,0,0,0]
    bestNumberOfFeatures = [0,0,0,0,0]

    bestFeaturesMask = [0]*5


    for k in range(1, x.shape[1]+1):
        skb = SelectKBest(chi2, k=k)
        skb.fit(x, y)
        X_new = skb.transform(x)
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = testPercent, random_state=0)

        plotModels(X_train, X_test, y_train, y_test)
        answers = models(X_train, X_test, y_train, y_test)

        for i in range(len(answers)):
                if answers[i] > bestTestAccuracy[i]:
                    bestTestAccuracy[i] = answers[i]
                    bestNumberOfFeatures[i] = k
                    bestFeaturesMask[i] = skb.get_support()

    return bestTestAccuracy,bestNumberOfFeatures,bestFeaturesMask

def plotModels(X_train, X_test, y_train, y_test):
    models = []
    models.append(DecisionTreeClassifier(max_depth=6, random_state = 42))
    models.append(AdaBoostClassifier(n_estimators=5, base_estimator=DecisionTreeClassifier(max_depth=6), learning_rate=0.001,random_state = 42))
    models.append(MLPClassifier(hidden_layer_sizes=32, learning_rate_init=0.01, random_state=0, solver='adam'))
    models.append(KNeighborsClassifier(n_neighbors=8, algorithm='brute'))
    models.append(SVC(probability=True))

    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
    for i in models:
        i.fit(X_train, y_train)
        yproba = i.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)
        result_table = result_table.append({'classifiers': i.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)
    result_table.set_index('classifiers', inplace=True)
    fig = plt.figure(figsize=(8, 6))
    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()

    # print(modeltest)
    # modeltest[0].plot()
    # modeltest[1].plot()
    # modeltest[2].plot()


    plt.show()

    # rfc = RandomForestClassifier(random_state=42)
    # rfc.fit(X_train, y_train)


if __name__ == '__main__':

        x, y = data_process()

        bestOfModels = []



        # base model
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        modelsWithAllFeatures = models(X_train, X_test, y_train, y_test)

        # before feature selection
        plotModels(X_train, X_test, y_train, y_test)

        for j in [x * 0.1 for x in range(1, 10)]:
            X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=j,random_state=0)

            bestOfModels.append(models(X_train, X_test, y_train, y_test))



        bestTestAccuracyAfterFS, bestNumberOfFeatures, bestFeaturesMask = modelsFeatureSelection(x, y, 0.2)


        best = pd.DataFrame(bestOfModels, columns = ['Decision tree', 'AdaBoost','MLP', 'KNN', 'SVM'])
        best.index = [x * 0.1 for x in range(1, 10)]


        # printing test size from 0.1-1
        disDF(best)

        # best.describe(include='all').T.style.background_gradient(subset=['mean', 'std', '50%', 'count'], cmap='RdPu')
        print(best.describe())


        # how come some features without feature selection get same score? KNN...
        print('Models With all the Features:')
        print("Decision Tree:\t\t Adaboost:\t\t\tMLP:\t\t\t\t KNN:\t\t\t\tSVM:")
        print(modelsWithAllFeatures)

        print('Models after feature selection:')
        print("Decision Tree:\t\t Adaboost:\t\t\tMLP:\t\t\t\t KNN:\t\t\t\tSVM:")
        print(bestTestAccuracyAfterFS)


        print(bestNumberOfFeatures)

        featuresSelected = pd.DataFrame(bestFeaturesMask, columns = ['Age', 'Sex','ChestPainType', 'RestingBP', 'Cholesterol','FastingBS', 'RestingECG','MaxHR', 'ExerciseAngina', 'Oldpeak','ST_Slope'])
        featuresSelected.index = ['Decision tree', 'AdaBoost','MLP', 'KNN', 'SVM']
        disDF(featuresSelected.transpose())


        # # after feature selection
        # plotModels(bestTestAccuracyAfterFS)