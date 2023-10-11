import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

nltk.download("stopwords")


def generate_keywords_using_tokens(token_list: list, keyword_amount: int = 10) -> dict[str, int]:
    """
    creates a list of dictionaries tracking every word in a token and its frequency of appearances \n
    to generate a list of words, you will have to use SpacyHandler from
    :param token_list: the list of words you will be using
    NOTE: you should be passing in a list that has every occurrence of a token
    :param keyword_amount: how many keywords you want to get back
    :return: a dictionary with the most frequent words
    """
    stop_words = set(stopwords.words('english'))
    # removes numbers and stop words
    token_list = [token for token in token_list if token.isalnum() and token not in stop_words]

    freq_tup = FreqDist(token_list).most_common(keyword_amount)

    # converts the tuple into a dictionary
    output_dict = {}
    for tup in freq_tup:
        word = tup[0]
        freq = tup[1]
        output_dict[word] = freq
    return output_dict

