from scipy.special import softmax


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
        "roberta_neg": scores[0],
        "roberta_neu": scores[1],
        "roberta_pos": scores[2]
    }
    return scores_dict
