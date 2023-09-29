# pip install -U pip setuptools wheel
# pip install -U spacy
# python -m spacy download en_core_web_sm
import spacy
# import tqdm
# for progress bar
from tqdm import tqdm

""" POS TAGS
ADJ: Adjective (e.g., "big," "red").
ADP: Adposition (e.g., "in," "to," "during").
ADV: Adverb (e.g., "quickly," "very").
AUX: Auxiliary verb (e.g., "is," "has").
CONJ: Conjunction (e.g., "and," "but").
CCONJ: Coordinating conjunction (e.g., "and," "or").
DET: Determiner (e.g., "a," "an," "the").
INTJ: Interjection (e.g., "oh," "hey").
NOUN: Noun (e.g., "cat," "house").
NUM: Numeral (e.g., "one," "3").
PART: Particle (e.g., "not," "to").
PRON: Pronoun (e.g., "he," "she," "it").
PROPN: Proper noun (e.g., "John," "Paris").
PUNCT: Punctuation (e.g., ".", ",", "-").
SCONJ: Subordinating conjunction (e.g., "if," "while").
SYM: Symbol (e.g., "$," "%").
VERB: Verb (e.g., "run," "eat").
X: Other (e.g., "xfm," "zzz").
"""


class SpacyHandler:
    def __init__(self, to_load="en_core_web_md"):
        self.nlp = spacy.load(to_load)

    def create_list_of_words(self,
                             to_include: str,
                             text_list: list[str],
                             repeat: bool = True) -> list[str]:
        """
        pulls words from a list of strings whose part of speech tags are in the pos_tags_to_include list
        :param text_list: a list of strings
        :param repeat: True = add text to output list even if they are already in the list
        :param to_include: a string of word types you want to pull separated by commas.
        e.g. "ADJ,ADV,NOUN,VERB" pulls adjectives, adverbs, nouns and verbs
        Full list of pos tags can be found in analyzer_spacy.py
        :return: returns words as a list of strings
        """
        list_of_words = []
        pos_tags_to_include = to_include.split(',')
        for i in tqdm(range(len(text_list)), desc=f"Extracting {pos_tags_to_include} from text"):
            word_lower = text_list[i].lower()
            # seperates the text into tokens (individual words)
            tokens = self.nlp(word_lower)
            for token in tokens:
                # check if the current word should be included in the output list
                if token.pos_ in pos_tags_to_include:
                    try:
                        text = token.text
                        # if the word is not in the list, OR you want to repeat
                        if text not in list_of_words or repeat:
                            list_of_words.append(text)
                    except Exception as e:
                        print(e, token)
        return list_of_words

    def load(self, to_load: str = "en_core_web_md"):
        """
        loads the spacy model you want to use
        to install use "python -m spacy download en_core_web_md" in terminal
        "en_core_web_md" can be changed to different sized models
        en_core_web_sm - small sized model. Efficient but not as accurate
        en_core_web_md - medium-sized model. More accurate than the small sized model
        en_core_web_lg - large sized model. Most accurate, but close to 1gb in size
        :param to_load: the name of the model you want to load
        """
        self.nlp = spacy.load(to_load)

    def create_list_of_verbs_lemma(self, text_list: list[str], repeat: bool = True) -> list[str]:
        """
        creates a list of strings that represent verbs that appeared in the text then converts them into dictionary form
        :param text_list: a list of strings
        :param repeat: True = add text to output list even if they are already in the list
        :return: returns dictionary form verbs as a list of strings
        """
        # gets list of verbs
        word_list = self.create_list_of_words(text_list=text_list,
                                              repeat=repeat,
                                              to_include="VERB")
        output_list = []
        # go through entire list and converts each word to dictionary form
        for i in tqdm(range(len(word_list)), desc="Converting verbs to dictionary form"):
            doc = self.nlp(word_list[i])
            output_list.append(doc[0].lemma_)
        return output_list

    def create_list_of_named_entities(self,
                                      text_list: list[str],
                                      repeat: bool = True) -> list[str]:
        """
        using spacy's entities, generate a list of named entites (like proper nouns)
        note that this data will be sorted by first appearance
        :param text_list: a list of strings
        :param repeat: True = add text to output list even if they are already in the list
        :return: returns word/phrases as a list of strings
        """
        output_list = []
        for i in tqdm(range(len(text_list)), desc=f"Extracting proper nouns from text"):
            text = text_list[i]
            # turning the text into a doc object
            doc = self.nlp(text)
            ents = doc.ents
            # we go through each identified entity in the list
            for ent in ents:
                # if you want to add repeated strings, then you do need to bother doing any check
                # if the entity is not already in the output list then add it
                # also check if the entities part of speech tag should be ignored
                if repeat or ent.text not in output_list:
                    output_list.append(ent.text)
                    # print(ent.text, pos_tag)
        return output_list

    def create_list_of_noun_chunks(self, text_list: list[str], repeat: bool = True) -> list[str]:
        """
        get nouns that have adjectives or verbs modifying them
        :param text_list: a list of strings
        :param repeat: True = add text to output list even if they are already in the list
        :return: returns word/phrases as a list of strings
        """
        output_list = []
        for i in tqdm(range(len(text_list)), desc=f"Extracting noun chunks from text"):
            text = text_list[i]
            # turning the text into a doc object
            doc = self.nlp(text)
            noun_chunks = doc.noun_chunks
            # go through each noun chunk found in the sentence
            for noun_chunk in noun_chunks:
                if repeat or noun_chunk.text not in output_list:
                    output_list.append(noun_chunk.text)
        return output_list

    def get_pos_tag(self, text: str) -> str:
        """
        returns a given words part of speech tag
        it works, but it doesn't seem accurate
        :param text: must be a single word (no phrase)
        :return:  pos tag as a string
        """
        return self.nlp(text)[0].pos_
