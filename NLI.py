import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt

def train_model(args):
    clf, X, y = args
    return clf.fit(X, y)

def predict_model(args):
    model, X = args
    return model.predict(X)

def train_and_predict(num_cores, X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Use a list of arguments for training
    training_args = [(clf, X_train, y_train)] * num_cores

    start_time = time.time()
    # Parallelize the training using concurrent.futures
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        trained_models = list(executor.map(train_model, training_args))
    end_time = time.time()
    elapsed_time_training = end_time - start_time

    # Use a list of arguments for prediction
    prediction_args = [(model, X_test) for model in trained_models]

    start_time = time.time()
    # Parallelize the prediction using concurrent.futures
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        predictions = list(executor.map(predict_model, prediction_args))
    end_time = time.time()
    elapsed_time_prediction = end_time - start_time

    # Combine predictions from different models
    final_predictions = sum(predictions) / len(predictions)

    # Evaluate the model
    accuracy = accuracy_score(y_test, final_predictions)

    return elapsed_time_training, elapsed_time_prediction, accuracy

def plot_speedup_efficiency(cores, training_times, prediction_times):
    sequential_time = training_times[0] + prediction_times[0]

    speedup = [sequential_time / (training + prediction) for training, prediction in zip(training_times, prediction_times)]
    efficiency = [s / c for s, c in zip(speedup, cores)]

    plt.plot(cores, speedup, marker='o', label='Speedup')
    plt.plot(cores, efficiency, marker='o', label='Efficiency')
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup/Efficiency')
    plt.title('Speedup and Efficiency vs. Number of Cores')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('C:/Users/cyber/Downloads/project/project/train.csv')
    selected_columns = ['premise', 'hypothesis', 'label']
    data = data[selected_columns]

    # Sample 10% of the dataframe
    data = data.sample(frac=0.1, random_state=42)

    # Drop rows with missing values
    data = data.dropna()

    # Split the dataset into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Tokenization and feature extraction
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_data['premise'] + ' ' + train_data['hypothesis'])
    y_train = train_data['label']

    # Tokenization and feature extraction for test data
    X_test = vectorizer.transform(test_data['premise'] + ' ' + test_data['hypothesis'])
    y_test = test_data['label']

    cores = [1, 2, 4, 8]
    training_times = []
    prediction_times = []
    accuracies = []

    for num_cores in cores:
        training_time, prediction_time, accuracy = train_and_predict(num_cores, X_train, y_train, X_test, y_test)
        training_times.append(training_time)
        prediction_times.append(prediction_time)
        accuracies.append(accuracy)
        print(f'Training with {num_cores} cores took {training_time} seconds.')
        print(f'Prediction with {num_cores} cores took {prediction_time} seconds.')
        print(f'Model Accuracy for {num_cores} cores: {accuracy}')
        print('---')

    # Plot speedup and efficiency
    plot_speedup_efficiency(cores, training_times, prediction_times)
