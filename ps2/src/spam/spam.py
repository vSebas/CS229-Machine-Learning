import collections

import numpy as np

import util


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    message = message.lower()
    words = message.split(" ")

    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    word_count = collections.defaultdict(int)
    for message in messages:
        words = set(get_words(message))
        for word in words:
            word_count[word] += 1 # Count in how many messages the word appears
    dictionary = {}
    
    index = 0
    for word, count in word_count.items():
        if count >= 5:
            dictionary[word] = index # Assign index to the word if it appears in at least 5 messages
            index += 1

    return dictionary
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    num_messages = len(messages)
    num_words = len(word_dictionary)
    matrix = np.zeros((num_messages, num_words), dtype=int)

    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word in word_dictionary:
                j = word_dictionary[word]
                matrix[i, j] += 1 # Increment the count for the word in the message
    
    return matrix
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    num_messages, num_words = matrix.shape
    nb_model = {}
    # Prior probabilities
    nb_model['prior_spam'] = np.sum(labels) / num_messages
    nb_model['prior_ham'] = 1 - nb_model['prior_spam']
    # Likelihoods with Laplace smoothing
    spam_word_counts = np.sum(matrix[labels == 1], axis=0) + 1
    ham_word_counts = np.sum(matrix[labels == 0], axis=0) + 1
    total_spam_words = np.sum(spam_word_counts) + num_words
    total_ham_words = np.sum(ham_word_counts) + num_words
    nb_model['likelihood_spam'] = spam_word_counts / total_spam_words
    nb_model['likelihood_ham'] = ham_word_counts / total_ham_words
    
    return nb_model

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    num_messages = matrix.shape[0]
    predictions = np.zeros(num_messages, dtype=int)
    log_prior_spam = np.log(model['prior_spam'])
    log_prior_ham = np.log(model['prior_ham'])
    log_likelihood_spam = np.log(model['likelihood_spam'])
    log_likelihood_ham = np.log(model['likelihood_ham'])
    for i in range(num_messages):
        log_prob_spam = log_prior_spam + np.sum(matrix[i] * log_likelihood_spam) # Compute log-probability for spam
        log_prob_ham = log_prior_ham + np.sum(matrix[i] * log_likelihood_ham)
        if log_prob_spam > log_prob_ham:
            predictions[i] = 1
        else:
            predictions[i] = 0

    return predictions

    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    scores = {}
    for word, index in dictionary.items():
        prob_word_given_spam = model['likelihood_spam'][index]
        prob_word_given_ham = model['likelihood_ham'][index]
        score = prob_word_given_spam / prob_word_given_ham # indicative score
        scores[word] = score
    sorted_words = sorted(scores, key=scores.get, reverse=True)
    top = sorted_words[:5]
    
    return top
    
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

if __name__ == "__main__":
    main()
