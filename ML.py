
"""
Naiwny klasyfikator Bayesa

System rekomendowania filmów
X - macierz filmów (ID - użytkownik, 0-nie podobał się, 1-podobał się), Y - macierz nowego filmu, czy się podbał (Y) czy nie (N)
ID  f1  f2  f3
1   0   1   1
2   0   0   1
3   0   0   0
4   1   1   0

X_test - próbka testowa

Program ma przewidzieć szansę z jaką nowy film będzię się podobał
"""

import numpy as np 
X_train = np.array([
    [0,1,1],
    [0,0,1],
    [0,0,0],
    [1,1,0]])
Y_train = ['Y','N','Y','Y']
X_test = np.array([[1,1,0]])

def get_label_indices(labels):
    """
Grupowanie próbek na podstawie ich etykiet i zwrócenie indeksów
@param labels: lita etykiet
@return: słownik, {klasa1: [indeksy], klasa2:[indeksy]}
    """
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index,label in enumerate(labels):
        label_indices[label].append(index)
    return label_indices


def get_prior(label_indices):
    """
    Wyliczanie prawdopodobieństwa  apriori na podstawie próbek treningowych
    @param label_indices: indeksy pogrupowane według klas
    @return: słownik, w którym kluczem jest etykieta klasy, a wartością 
    odpowiednie prawdopodobieństwo apriori
    """

    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= total_count
    return prior

def get_likelihood(features, label_indices, smoothing=0):
    """
    Wyliczenie szansy na podstawie próbek trenngowych

    """
    likelihood = {}
    for label, indices in label_indices.items():
        likelihood[label] = features[indices,:].sum(axis=0) + smoothing
        total_count = len(indices)
        likelihood[label] = likelihood[label] / (total_count + 2*smoothing)
    return likelihood


def get_posterior(X, prior, likelihood):
    """
    Wyliczanie prawdopodobieństwa a posteriori na podstawie a priori i szansy
    @param X: próbki treningowe
    @param prior: słownik, w którym jest etykieta klasy, a wartością odpowiednie
    prawdopodobieństwo a priori
    @return: słownik, w którym kluczem jest etykieta klasy, a wartością odpowiednie
    prawdopodobieństwo a posteriori
    """

    posteriors = []
    for x in X:
        #A posteriori jest proporcjonalne do a priori *  szansa
        posterior = prior.copy()
        for label, likelihood_label, in likelihood.items():
            for index, bool_value in enumerate(x):
                posterior[label] *= likelihood_label[index] if bool_value else (1-likelihood_label[index])
    #normalizacja aby wszystko zsumowało się do 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

label_indices = get_label_indices(Y_train)
print('label_indices:\n', label_indices)

prior = get_prior(label_indices)
print('A priori:', prior)

smoothing = 1
likelihood = get_likelihood(X_train,label_indices,smoothing)
print('Szansa:\n', likelihood)

posterior = get_posterior(X_test, prior, likelihood)
print('A posteriori:\n', posterior)
