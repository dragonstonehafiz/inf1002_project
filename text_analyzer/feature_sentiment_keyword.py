from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import sys

from class_SpacyHandler import SpacyHandler
from func_roberta import polarity_scores_roberta
from func_keyword_gen import generate_keywords_using_tokens

spacy_handler = SpacyHandler()


def generate_sentiment(comment_list: list[str]) -> list[dict]:
    """
    creates a list of dictionaries that tracks a comments polarity scores \n
    the dictionaries will not contain the text's id \n
    dictionary structure: {"roberta_neg", "roberta_neu", "roberta_pos"} \n
    this function includes a list of ids, so I can merge the output with the original data
    :param comment_list: list of strings of the comments
    :return: returns a list containing dictionaries
    """
    # Found on
    # https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    except ImportError as e:
        print("PyTorch is not installed on this environment\n"
              "It is necessary for sentiment analysis\n"
              "Please install it using 'pip install torch'")
        sys.exit()

    output_list = []
    # generate sentiment of each individual text then make a dictionary with all that data
    # all that data goes into a list which is outputted
    for i in tqdm(range(len(comment_list)), desc=f"Analyzing Comments"):
        text = comment_list[i]
        try:
            result = polarity_scores_roberta(text, tokenizer, model)
        # if sentiment analysis fails input default values
        except Exception as e:
            print(f"{text[:100]}... unable to be analyzed.\n"
                  f"ERROR: {e}\n"
                  f"Inputting failure values")
            result = {"roberta_neg": -1, "roberta_neu": -1, "roberta_pos": -1}
        output_list.append(result)
    return output_list


def generate_keywords(text_list: list[str], keyword_amount: int) -> dict:
    """
    creates a list of most frequently used words
    :param text_list: a list of all the text you want to find keywords for
    :param keyword_amount: the amount of keywords you want to get back
    :return: a dictionary where the key is the word and the data is its frequency
    """
    word_list = spacy_handler.create_list_of_words(to_include="",
                                                   text_list=text_list,
                                                   repeat=True)
    keywords = generate_keywords_using_tokens(word_list, keyword_amount)
    return keywords


def generate_keywords_adjectives(text_list: list[str], keyword_amount: int) -> dict:
    """
    creates a list of most frequently used adjectives
    :param text_list: a list of all the text you want to find keywords for
    :param keyword_amount: the amount of keywords you want to get back
    :return: a dictionary where the key is the word and the data is its frequency
    """

    word_list = spacy_handler.create_list_of_words(to_include="ADJ",
                                                   text_list=text_list,
                                                   repeat=True)
    keywords = generate_keywords_using_tokens(word_list, keyword_amount)
    return keywords


def generate_keywords_verbs(text_list: list[str], keyword_amount: int) -> dict:
    """
    creates a list of most frequently used verbs
    :param text_list: a list of all the text you want to find keywords for
    :param keyword_amount: the amount of keywords you want to get back
    :return: a dictionary where the key is the word and the data is its frequency
    """
    word_list = spacy_handler.create_list_of_words(to_include="VERB",
                                                   text_list=text_list,
                                                   repeat=True)
    keywords = generate_keywords_using_tokens(word_list, keyword_amount)
    return keywords


def generate_keywords_nouns(text_list: list[str], keyword_amount: int) -> dict:
    """
    creates a list of most frequently used nouns and proper nouns
    :param text_list: a list of all the text you want to find keywords for
    :param keyword_amount: the amount of keywords you want to get back
    :return: a dictionary where the key is the word and the data is its frequency
    """
    word_list = spacy_handler.create_list_of_words(to_include="NOUN,PROPN",
                                                   text_list=text_list,
                                                   repeat=True)
    keywords = generate_keywords_using_tokens(word_list, keyword_amount)
    return keywords


def generate_list_of_words(text_list: list[str]) -> list[str]:
    """
    returns a list of words from a given text, including every appearance of each word
    :param text_list: the list of text you want to pull words from
    :return: a list of all words in that appeared in the text (including repeated appearances)
    """
    word_list = spacy_handler.create_list_of_words(to_include="",
                                                   text_list=text_list,
                                                   repeat=True)
    return word_list


def generate_list_of_adjectives(text_list: list[str]) -> list[str]:
    """
    returns a list of adjectives from a given text, including every appearance of each word
    :param text_list: the list of text you want to pull words from
    :return: a list of all adjectives in that appeared in the text (including repeated appearances)
    """
    word_list = spacy_handler.create_list_of_words(to_include="ADJ",
                                                   text_list=text_list,
                                                   repeat=True)
    return word_list


def generate_average_polarity_scores(name_list: list[str],
                                     neg_list: list[float],
                                     neu_list: list[float],
                                     pos_list: list[float]) -> list[dict]:
    """
    generate averages of polarity scores separated by types of shoes
    :param name_list: a list of strings with the names of all entries
    :param neg_list: a list of floats with the negative scores of all entries
    :param neu_list: a list of floats with the neutral scores of all entries
    :param pos_list: a list of floats with the positive scores of all entries
    :return: returns a list with dictionaries of polarity scores
    dictionary structure = {"Shoe Name", "Roberta Negative", "Roberta Neutral", "Roberta Positive"}
    """
    # This loop is to generate the sum of all values in for each polarity score
    sum_dict = {}
    for i in range(0, len(name_list)):
        shoe_name = name_list[i]
        neg_score = neg_list[i]
        neu_score = neu_list[i]
        pos_score = pos_list[i]
        # checks the dictionary to see if there is a key tracking the current shoe
        shoe_data_list = sum_dict.get(shoe_name)
        if shoe_data_list is None:
            # since the map doesn't already have a list tracking scores, we have to initialize a new list
            # index = 3 is total number of entries
            sum_dict[shoe_name] = [neg_score, neu_score, pos_score, 1]
        else:
            # since the map already has a list, add new data to list
            shoe_data_list[0] += neg_score
            shoe_data_list[1] += neu_score
            shoe_data_list[2] += pos_score
            shoe_data_list[3] += 1
    # calculate the average for each polarity score for each unique shoe
    output_list = []
    # key value pair, just renamed them for clarity
    for shoe_name, polarity_scores in sum_dict.items():
        new_dict = {
            "Shoe Name": shoe_name,
            "Roberta Negative": polarity_scores[0]/polarity_scores[3],
            "Roberta Neutral": polarity_scores[1]/polarity_scores[3],
            "Roberta Positive": polarity_scores[2]/polarity_scores[3]
        }
        output_list.append(new_dict)

    return output_list
