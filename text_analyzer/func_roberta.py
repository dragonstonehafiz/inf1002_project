from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm


def polarity_scores_roberta(text: str, tokenizer, model) -> dict:
    """
    find the polarity score of a text using RoBERTa pretrained model
    {"roberta_neg", "roberta_neu", "roberta_pos"} all values in this are from zero to one and all three add together
    will add up to one
    :param text: the text you want to find the sentiment for
    :param tokenizer: auto tokenizer object
    :param model: the loaded model
    :return: a dictionary of the text's sentiment score {"roberta_neg", "roberta_neu", "roberta_pos"}
    """
    # Run for Roberta Model
    encoded_text = tokenizer(text, return_tensors='pt')
    output = model(**encoded_text)
    # turn into numpy to store locally
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict


def generate_sentiment_with_id(comment_list: list[str], id_list: list[str], id_col_name: str = "id") -> list[dict]:
    """
    creates a list of dictionaries that tracks a comments polarity scores \n
    this dictionary will contain the text's id \n
    dictionary structure: {"id"," roberta_neg", "roberta_neu", "roberta_pos"} \n
    this function includes a list of ids, so I can merge the output with the original data
    :param comment_list: list of strings of the comments
    :param id_list: list of strings of the comment ids
    :param id_col_name: the name of the column that will store item ids
    :return: returns a list containing dictionaries
    """
    # Found on
    # https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    output_list = []
    for i in tqdm(range(len(comment_list)), desc=f"Analyzing Comments"):
        try:
            text = comment_list[i]
            comment_id = id_list[i]
            result = {**{id_col_name: comment_id}, **polarity_scores_roberta(text, tokenizer, model)}
            output_list.append(result)
        except Exception as e:
            print(f"{comment_id} unable to be analyzed.\n"
                  f"ERROR: {e}\n"
                  f"Inputting failure values")
            text = comment_list[i]
            comment_id = id_list[i]
            result = {'id': comment_id, "roberta_neg": -1, "roberta_neu": -1, "roberta_pos": -1}
            output_list.append(result)
    return output_list




