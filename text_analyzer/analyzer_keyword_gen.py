import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def keyword_gen_using_sentences(text_list: list[str], keyword_amount: int = 10) -> list[dict]:
    """
    generates a list of keywords using sklearn's CountVectorizer
    :param text_list: a list of all the text you want to check
    :param keyword_amount: the amount of keywords you want returned back
    :return: a list of dictionaries {"word", "freq"}
    NOTE freq isn't word count. Not sure how to get that working with count vectorizer
    """
    # max_df means if a word appears in more than 90 percent of the document, it will be ignored
    # min_df means a word needs to appear at least twice need to be tracked
    cv = CountVectorizer(stop_words="english", max_df=0.9, min_df=2, strip_accents='ascii')
    # counts word use and stores them in a matrix
    word_counts = cv.fit_transform(text_list)
    # word_counts has one row for each document and one column for each word
    # (0, 5) 1
    # means WORD 5 appears in DOCUMENT 0 1 time

    # a list of words that appeared in the text
    words = cv.get_feature_names_out()

    transformer = TfidfTransformer()
    transformer.fit(word_counts)
    # get the word count vector for the first comment
    word_count_vector = cv.transform(text_list)
    # create tfidf vector
    tfidf_vector = transformer.transform(word_count_vector).tocoo()
    # print(tfidf_vector)

    tuples = zip(tfidf_vector.row, tfidf_vector.col, tfidf_vector.data)
    # sort text by value in at item 3 of tuple (freq of appearances)
    tfidf_vector = sorted(tuples, key=lambda x: x[2], reverse=True)
    # return the specified amount of keywords
    top_keywords = tfidf_vector[:keyword_amount]
    output_list = []
    for tup in top_keywords:
        # tup[1] is the words index in the word vector
        new_dict = {"word": words[tup[1]],
                    # "freq: cv.vocabulary_[words[tup[1]]]
                    "freq": tup[2]}
        output_list.append(new_dict)
    return output_list


def keyword_gen_using_tokens(token_list: list, keyword_amount: int = 10) -> list[dict[str, int]]:
    """
    creates a list of dictionaries tracking every word in a token and its frequency of appearances
    :param token_list: the list of words you will be using
    NOTE: you should be passing in a list that has every occurrence of a token
    :param keyword_amount: how many keywords you want to get back
    :return: a list of tuples the size of keyword_amount
    """
    nltk.download("stopwords")
    stop_words = set(stopwords.words('english'))
    # removes numbers and stop words
    token_list = [token for token in token_list if token.isalnum() and token not in stop_words]

    freq_dist = FreqDist(token_list)
    return convert_most_common_tuple_to_list(freq_dist.most_common(keyword_amount))


def convert_most_common_tuple_to_list(list_of_tuples: list[tuple[str, int]]) -> list[dict[str, int]]:
    """
    creates a list of dictionaries tracking keyword and frequency of appearances
    :param list_of_tuples: the list of tuples generated from freq_dist.most_common
    :return: a list of dictionaries {"word", "freq"}
    """
    output_list = []
    for tup in list_of_tuples:
        word = tup[0]
        freq = tup[1]
        new_dict = {"word": word,
                    "freq": freq}
        output_list.append(new_dict)
    return output_list
