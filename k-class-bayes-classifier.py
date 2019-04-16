import sys
import numpy as np

def class_priors(y_train, counts):
    prior = []
    total = len(y_train)
    for i in counts:
        prior.append(i / total)
    return prior

def class_conditional_density(X_train, y_train, k_classes, counts):
    mean = []
    covariance = []
    for i, j in zip(k_classes, counts):
        l_class = y_train == i
        mean.append(j / len(X_train[l_class]))
        covariance.append(np.dot(
            (X_train[l_class] - mean[-1]).T, (X_train[l_class] - mean[-1])) / len(X_train[l_class]))
    return mean, covariance

def pluginClassifier(X_train, y_train, X_test):
    k_classes, counts = np.unique(y_train, return_counts=True)
    prior = class_priors(y_train, counts)
    mean, covariance = class_conditional_density(
        X_train, y_train, k_classes, counts)
    posterior = []
    convinv = np.linalg.inv(covariance)
    convisqrt = 1 / np.sqrt(np.linalg.det(covariance))
    for x in X_test:
        for i in range(len(prior)):
            probability = convisqrt[i] * np.exp(-0.5 * np.dot(
                x - mean[i], np.dot(convinv[i], (x - mean[i]).T)))
            posterior.append(prior[i] * probability)
    posterior_prob = np.reshape(posterior, (X_test.shape[0], len(k_classes)))
    s = np.reshape(np.sum(posterior_prob, axis=1), (-1, 1))
    posterior_prob = posterior_prob / s
    return posterior_prob

def main():
    X_train = np.genfromtxt(sys.argv[1], delimiter=",")
    y_train = np.genfromtxt(sys.argv[2])
    X_test = np.genfromtxt(sys.argv[3], delimiter=",")

    final_outputs = pluginClassifier(X_train, y_train, X_test)

    y = final_outputs.argmax(1)
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")

if __name__ == "__main__":
    main()
