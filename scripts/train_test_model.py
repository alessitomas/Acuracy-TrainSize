import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import json
from tqdm import tqdm




def readStorage(storageJsonPath):
    with open(storageJsonPath, 'r') as f:
        storage = json.load(f)
        return storage


# função para criar métricas, testando e avaliando o modelo com diferente quantidade de dados

def performanceOfTrainSizes(X, y):
    pipeline = Pipeline([
                            ('StandardScaler',  StandardScaler()),
                            ('meu_classificador', LogisticRegression(max_iter=10000))
                        ])

    n_train = np.logspace(-2, np.log10(0.8), 10)
    n_train = [int(n*X.shape[0]) for n in n_train]
    n_train = [n for n in n_train if n > len(set(y))]
    acc_mean = []
    acc_std = []
    acc_mean_overfit = []
    acc_std_overfit = []
    
    i = 0
    for n in n_train:

        this_acc = []
        this_acc_overfit = []
        for k in tqdm(range(50)):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, stratify=y)
            
            # accurracy
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_pred,y_test)
            # acurracy overfit

            y_pred_overfit = pipeline.predict(X_train)
            acc_overfit = accuracy_score(y_pred_overfit,y_train)

            this_acc_overfit.append(acc_overfit)
            this_acc.append(acc)
        
        this_acc_overfit = np.array(this_acc_overfit)
        this_acc = np.array(this_acc)
        acc_mean.append(this_acc.mean())
        acc_std.append(this_acc.std())
        acc_mean_overfit.append(this_acc_overfit.mean())
        acc_std_overfit.append(this_acc_overfit.std())
        i += 1
    return acc_mean, acc_mean_overfit, acc_std, acc_std_overfit, n_train

# Função para salvar métricas

def saveMetricsStorage(storageJsonPath ,datasetName, acc_mean, acc_mean_overfit, acc_std, acc_std_overfit, n_train,datasetSize):
    with open(storageJsonPath, 'r') as f:
        storage = json.load(f)
    
    storage[datasetName] = {
        'acc_mean': acc_mean,
        'acc_mean_overfit': acc_mean_overfit,
        'acc_std': acc_std,
        'acc_std_overfit': acc_std_overfit,
        'n_train_': n_train,
        'datasetSize': datasetSize
    }
    
    with open(storageJsonPath, 'w') as f:
        json.dump(storage, f)



# PIPELINE-FMA
# constantes para determinar número de dados para treino
n_train = np.logspace(-2, np.log10(0.8), 10)
fma_subsets = ['fma-small', 'fma-medium', 'fma-large','fma-small-handcrafted', 'fma-medium-handcrafted', 'fma-large-handcrafted']
intersaction_tack_ids = readStorage('../data/intersection-trackIds.json')

for fma_subset in tqdm(fma_subsets):
    datapath = f"../data/fma/{fma_subset}.csv"
    dataset = pd.read_csv(datapath)
    intersaction_tack_ids_subset = intersaction_tack_ids[fma_subset.removesuffix('-handcrafted')]
    dataset = dataset[dataset['track_id'].isin(intersaction_tack_ids_subset)]
    print(len(dataset))

    X = dataset.drop(['track_id','genre_top','subset'], axis=1)
    y = dataset['genre_top']
    acc_mean, acc_mean_overfit , acc_std, acc_std_overfit ,n_train = performanceOfTrainSizes(X, y)
    saveMetricsStorage('../data/metrics.json', fma_subset, acc_mean, acc_mean_overfit , acc_std, acc_std_overfit ,n_train,len(y))