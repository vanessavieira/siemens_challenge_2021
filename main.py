import pandas as pd


def read_data(filename):
    data = pd.read_excel(filename)
    return data


def analyse_data_with_plots(data):
    data = data


def split_data(data):
    train_data = data
    test_data = data
    return train_data, test_data


def train(data):
    data = data


def test(data):
    data = data


def run_model(data):
    data = data


def compute_probability_error_upper_bound(data):
    data = data


def analyse_results(data):
    data = data


if __name__ == '__main__':
    # Read data
    dataset = read_data('trainingdata_a')

    # Split data
    # Plot data
    # Train model with train data
    # Run model with test data
    # Compute Chernoff bound for probability error
    # Analyse results
