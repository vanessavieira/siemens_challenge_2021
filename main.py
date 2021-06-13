import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB


def read_data(filename):
    data = pd.read_excel(filename, index_col=0)
    print(data)
    return data


def analyse_data_with_plots(data):
    plt.scatter(x=data.x_i1, y=data.x_i2, c=data.l_i.astype('category').cat.codes,
                cmap=colors.ListedColormap(["red", "green"]))
    plt.show()


def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42, shuffle=True)
    return train, test


def train_model(data):
    model = BernoulliNB()
    model.fit(data[["x_i1", "x_i2"]], data["l_i"])
    return model


def test_model(data, model):
    predict = model.predict(data[["x_i1", "x_i2"]])
    return predict


def compute_probability_error_upper_bound(data):
    data = data


def analyse_results(predicted_data, train, test, model):
    print('Model accuracy score: {0:0.4f}'. format(accuracy_score(test["l_i"], predicted_data)))
    print('Training set score: {:.4f}'.format(model.score(train[["x_i1", "x_i2"]], train["l_i"])))
    print('Test set score: {:.4f}'.format(model.score(test[["x_i1", "x_i2"]], test["l_i"])))

    cm = confusion_matrix(test["l_i"], predicted_data)
    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0, 0])
    print('\nTrue Negatives(TN) = ', cm[1, 1])
    print('\nFalse Positives(FP) = ', cm[0, 1])
    print('\nFalse Negatives(FN) = ', cm[1, 0])


if __name__ == '__main__':
    # Read data
    dataset = read_data('/dataset/trainingdata_c.xlsx')

    # Plot data
    # analyse_data_with_plots(dataset)

    # Split data
    train_data, test_data = split_data(dataset)
    # analyse_data_with_plots(train_data)
    # analyse_data_with_plots(test_data)

    # Train model with train data
    data_model = train_model(train_data)

    # Run model with test data
    predictions = test_model(test_data, data_model)

    # Analyse results
    analyse_results(predictions, train_data, test_data, data_model)

    # Compute Chernoff bound for probability error
