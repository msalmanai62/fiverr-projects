import sys
import math
from collections import defaultdict, Counter

# reading and preprocessing the text data file
def preprocess_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()
    return text

# unigram model 
def build_unigram_model(training_file):
    training_text = preprocess_text_file(training_file)
    words = training_text.split() # all words or unigrams in the text file
    total_words = len(words) # total number of unigrams
    word_counts = Counter(words) # make dictionary of words with counts
    return word_counts, total_words

# unigram model probability calculating function
def unigram_probability(sentence, word_counts, total_words):
    words = sentence.split() # split the sentance to words
    log_prob = 0
    for word in words:
        word_prob = word_counts[word] / total_words if word in word_counts else 0 # calculating the probabilty of each word in sentence
        log_prob += math.log2(word_prob) if word_prob > 0 else float('-inf') # adding the each words probability to total probability after applying log2
    return log_prob if log_prob!=float('-inf') else "undefined"
# bigram model
def build_bigram_model(training_file):
    training_text = preprocess_text_file(training_file)
    sentences = training_text.split('\n')
    bigram_counts = defaultdict(int) # initializing an empty dictionary for bigram counts
    word_counts = defaultdict(int) # initializing an empty dictionary for words/unigram counts
    
    for sentence in sentences:
        words = ['<s>'] + sentence.split() # adding special character <s> before each sentence
        for i in range(len(words) - 1): # 
            bigram = (words[i], words[i+1])  # making bigrams -> (bigram = current_word + next_word)
            bigram_counts[bigram] += 1 # bigram frequency calculation
            word_counts[words[i]] += 1 # unigram frequency calculation
            
    return bigram_counts, word_counts

# bigram probability calculation without smoothing 
def bigram_probability(sentence, bigram_counts, word_counts):
    words = ['<s>'] + sentence.split()
    log_prob = 0
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_prob = bigram_counts[bigram] / word_counts[words[i]] if bigram_counts[bigram] > 0 else 0
        log_prob += math.log2(bigram_prob) if bigram_prob > 0 else float('-inf')
    return log_prob if log_prob!=float('-inf') else "undefined"

# bigram probability calculation smoothing 
def bigram_add_one_smoothing(sentence, bigram_counts, word_counts, V): # V = len(unigram_counts_for_bigrams) + 1  # +1 for <s> token
    words = ['<s>'] + sentence.split()
    log_prob = 0
    for i in range(len(words) - 1):
        bigram = (words[i], words[i+1])
        bigram_prob = (bigram_counts[bigram] + 1) / (word_counts[words[i]] + V)
        log_prob += math.log2(bigram_prob)
    return log_prob
# models training and testing function
def test_models_with_file(training_file, test_file):
    """
    Test the specified language model with sentences from a test file.
    
    Parameters:
    - test_file: Path to the test text file.
    - training_file: Path to the training text file.
    """
    # unigram model
    word_counts, total_words = build_unigram_model(training_file)
    # bigram model
    bigram_counts, unigram_counts_for_bigrams = build_bigram_model(training_file)
    V = len(unigram_counts_for_bigrams)-1

    with open(test_file, 'r', encoding='utf-8') as file:
        sentences = file.readlines()

    for sentence in sentences:
        preprocessed_sentence = sentence.lower().strip()

        # unigram model probs
        log_prob_uni = unigram_probability(preprocessed_sentence, word_counts, total_words)

        # bigram probs
        log_prob_bi = bigram_probability(preprocessed_sentence, bigram_counts, unigram_counts_for_bigrams)

        # bigram probs with smoothing
        log_prob_bi_smoth = bigram_add_one_smoothing(preprocessed_sentence, bigram_counts, unigram_counts_for_bigrams, V)
        
        print(f"S = {sentence.strip()}")
        print(f"Unsmoothed Unigrams, logprob(S) = {round(log_prob_uni, 4) if log_prob_uni!='undefined' else log_prob_uni}")
        print(f"Unsmoothed Bigrams, logprob(S) = {round(log_prob_bi, 4) if log_prob_bi!='undefined' else log_prob_bi}")
        print(f"Smoothed Bigrams, logprob(S) = {log_prob_bi_smoth:.4f}")
        print() # add empty line 
# main function
def main():
    if len(sys.argv) != 3:
        print("Usage: python3 ngrams.py [training file] [test file]")
        sys.exit(1)
    
    training_file, test_file = sys.argv[1], sys.argv[2]
    test_models_with_file(training_file, test_file)

if __name__ == "__main__":
    main()