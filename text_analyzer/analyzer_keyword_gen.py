import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
# from rake_nltk import Rake


def keyword_gen_using_tokens(token_list: list, keyword_amount: int = 10) -> list[dict[str, int]]:
    """
    creates a list of dictionaries tracking keyword and frequency of appearances
    :param token_list: the list of words you will be using
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
