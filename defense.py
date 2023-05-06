import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error


def train_test_split(
    base_set,
    adversarial_set,
    percent_train=0.8,
    auto_balance=True,
    classification=True,
    column="logit_diffs",
    seed=0,
):
    """
    Splits the base and adversarial datasets into training and test sets.

    Args:
        base_set: DataFrame containing the base dataset.
        adversarial_set: DataFrame containing the adversarial dataset.
        percent_train: Float, percentage of the dataset used for training (default 0.8).
        auto_balance: Bool, if True, balances the training and test sets (default True).
        classification: Bool, if True, prepares data for classification task (default True).
        column: String, the column name containing features (default "logit_diffs").
        seed: Int, random seed for reproducibility (default 0).

    Returns:
        train_x, train_y, test_x, test_y: Lists containing training and test datasets.
    """
    np.random.seed(seed)

    # Shuffle the datasets
    base_set = base_set.sample(frac=1).reset_index(drop=True)
    adversarial_set = adversarial_set.sample(frac=1).reset_index(drop=True)

    # Calculate the training and test set sizes
    base_train_size = int(len(base_set) * percent_train)
    adversarial_train_size = int(len(adversarial_set) * percent_train)
    base_test_size = len(base_set) - base_train_size
    adversarial_test_size = len(adversarial_set) - adversarial_train_size

    # Balance the training and test set sizes if auto_balance is True
    if auto_balance:
        base_train_size = min(base_train_size, adversarial_train_size)
        base_test_size = min(base_test_size, adversarial_test_size)
        adversarial_train_size = base_train_size
        adversarial_test_size = base_test_size

    # Get training set
    base_train = base_set[column][:base_train_size]
    adversarial_train = adversarial_set[column][:adversarial_train_size]
    base_train = base_train[: int(len(base_train) / 2)]
    adversarial_train = adversarial_train[int(len(adversarial_train) / 2) :]

    # Concatenate and process the training set
    train_x = pd.concat([base_train, adversarial_train])
    try:
        train_x = [list(x) for x in train_x]
    except TypeError:
        pass

    # Prepare training labels
    if classification:
        train_y = [0] * len(base_train) + [1] * len(adversarial_train)
    else:
        train_y = (
            base_set["confidence"][: len(base_train)].tolist()
            + adversarial_set["confidence"][: len(adversarial_train)].tolist()
        )

    # Get test set
    base_test = base_set[column][len(base_train) :]
    adversarial_test = adversarial_set[column][len(adversarial_train) :]

    # Balance the test set if auto_balance is True
    if auto_balance:
        test_length = min(base_test_size, adversarial_test_size)
        base_test = base_test[:test_length]
        adversarial_test = adversarial_test[:test_length]

    # Print the number of examples in each set
    print(
        "Number of base train examples: {}, number of adversarial train examples: {}".format(
            len(base_train), len(adversarial_train)
        )
    )
    print(
        "Number of base test examples: {}, number of test adversarial examples: {}".format(
            len(base_test), len(adversarial_test)
        )
    )

    # Concatenate and process the test set
    test_x = pd.concat([base_test, adversarial_test])
    try:
        test_x = [list(x) for x in test_x]
    except TypeError:
        pass

    # Prepare test labels
    if classification:
        test_y = [0] * len(base_test) + [1] * len(adversarial_test)
    else:
        test_y = (
            base_set["confidence"][len(base_train) :].tolist()
            + adversarial_set["confidence"][len(adversarial_train) :].tolist()
        )

    return train_x, train_y, test_x, test_y


def train_defense_classifier(train_x, train_y, classification=True):
    if classification:
        # Train SVM classifier
        clf = svm.SVC()
    else:
        # Train SVM regressor
        clf = svm.SVR()

    clf.fit(train_x, train_y)

    return clf


def test_model(clf, test_x, test_y, classification=True, verbose=True):
    # Test SVM
    predictions = clf.predict(test_x)

    # Convert to numpy arrays
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    # Separate base and adversarial examples
    base_x = test_x[test_y == 0]
    base_y = test_y[test_y == 0]
    adversarial_x = test_x[test_y == 1]
    adversarial_y = test_y[test_y == 1]

    # Separate base and adversarial predictions
    base_predictions = predictions[test_y == 0]
    adversarial_predictions = predictions[test_y == 1]

    # Calculate overall accuracy or MSE
    overall_accuracy = sum(predictions == test_y) / len(test_y)

    # Print performance metrics
    if verbose:
        if classification:
            # Get accuracy
            try:
                base_accuracy = sum(base_predictions == base_y) / len(base_y)
                print("Base accuracy: {}".format(base_accuracy))
            except ZeroDivisionError:
                pass

            try:
                adversarial_accuracy = sum(
                    adversarial_predictions == adversarial_y
                ) / len(adversarial_y)
                print("Adversarial accuracy: {}".format(adversarial_accuracy))
            except ZeroDivisionError:
                pass
            print(
                "Overall accuracy: {}".format(sum(predictions == test_y) / len(test_y))
            )
        else:
            # Mean squared error
            base_mse = mean_squared_error(base_predictions, base_y)
            print("Base MSE: {}".format(base_mse))

            adversarial_mse = mean_squared_error(adversarial_predictions, adversarial_y)
            print("Adversarial MSE: {}".format(adversarial_mse))

            print("Overall MSE: {}".format(mean_squared_error(predictions, test_y)))

    return predictions, overall_accuracy


# Baseline using only first logit difference
def threshold_baseline(
    test_x, test_y, threshold, index=0, exclude_zeros=True, verbose=False
):
    """
    Computes the accuracy of a simple threshold-based defense using the specified threshold
    and index of the logit differences.

    Args:
        test_x: List of test examples.
        test_y: List of test labels.
        threshold: Float, the threshold value to be used for classification.
        index: Integer, the index of the logit difference to use (default 0).
        exclude_zeros: Bool, if True, exclude examples with zero logit difference (default True).
        verbose: Bool, if True, prints additional information (default False).

    Returns:
        accuracy: Float, the accuracy of the threshold-based defense.

    """
    correct_base = 0
    correct_adversarial = 0
    total_base = 0
    total_adversarial = 0
    for i in range(len(test_x)):
        if exclude_zeros and test_x[i][index] == 0:
            continue
        if test_x[i][index] <= threshold:
            if test_y[i] == 1:
                correct_adversarial += 1
                total_adversarial += 1
            else:
                total_base += 1
        else:
            if test_y[i] == 0:
                correct_base += 1
                total_base += 1
            else:
                total_adversarial += 1

    if verbose:
        if total_base == 0:
            print("No base examples")
        else:
            print("Base accuracy: {}".format(correct_base / total_base))

        if total_adversarial == 0:
            print("No adversarial examples")
        else:
            print(
                "Adversarial accuracy: {}".format(
                    correct_adversarial / total_adversarial
                )
            )
        print(
            "correct_base: {}, correct_adversarial: {}, total_base: {}, total_adversarial: {}".format(
                correct_base, correct_adversarial, total_base, total_adversarial
            )
        )

    return (correct_base + correct_adversarial) / (total_base + total_adversarial)


def get_best_threshold(train_x, train_y, index=0, abs_precision=0.0001):
    """
    Finds the best threshold for the threshold-based defense using a ternary search method.
    Args:
        train_x: List of training examples.
        train_y: List of training labels.
        index: Integer, the index of the logit difference to use (default 0).
        abs_precision: Float, the absolute precision for the search (default 0.0001).

    Returns:
        best_threshold: Float, the best threshold found.
    """
    # Ternary search for best threshold
    low = min([x[index] for x in train_x])
    high = max([x[index] for x in train_x])

    while abs(high - low) > abs_precision:
        low_third = low + (high - low) / 3
        high_third = high - (high - low) / 3

        if threshold_baseline(train_x, train_y, low_third, index) < threshold_baseline(
            train_x, train_y, high_third, index
        ):
            low = low_third
        else:
            high = high_third

    best_threshold = (low + high) / 2

    return best_threshold


def confidence_threshold(test_x, test_y, threshold):
    """
    Computes the accuracy of a simple threshold-based defense using the specified threshold

    Args:
        test_x: List or Pandas Series of confidence values for test examples.
        test_y: List of test labels.
        threshold: Float, the confidence threshold to use for predictions.

    Returns:
        accuracy: Float, the accuracy of predictions based on the confidence threshold.
    """
    correct_base = 0
    correct_adversarial = 0
    total_base = 0
    total_adversarial = 0
    for i in range(len(test_x)):
        if test_x.iloc[i] <= threshold:
            if test_y[i] == 1:
                correct_adversarial += 1
                total_adversarial += 1
            else:
                total_base += 1
        else:
            if test_y[i] == 0:
                correct_base += 1
                total_base += 1
            else:
                total_adversarial += 1

    return (correct_base + correct_adversarial) / (total_base + total_adversarial)


def get_best_confidence_threshold(train_x, train_y, abs_precision=0.0001):
    """
    Finds the best threshold for the threshold-based defense with confidence using a ternary search method.
    Args:
        train_x: List of training examples.
        train_y: List of training labels.
        abs_precision: Float, the absolute precision for the search (default 0.0001).
    Returns:
        best_threshold: Float, the best threshold found.
    """
    low = min(train_x)
    high = max(train_x)
    while abs(high - low) > abs_precision:
        # print(low, high)
        low_third = low + (high - low) / 3
        high_third = high - (high - low) / 3

        if confidence_threshold(train_x, train_y, low_third) < confidence_threshold(
            train_x, train_y, high_third
        ):
            low = low_third
        else:
            high = high_third

    best_threshold = (low + high) / 2

    return best_threshold
