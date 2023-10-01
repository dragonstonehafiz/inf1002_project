import pandas as pd
import sys
from analyzer_spacy import SpacyHandler
from analyzer_keyword_gen import keyword_gen_using_tokens, keyword_gen_using_sentences

""" 
PLEASE READ
No new files will be created here. We'll just print out top ten words to the terminal (using both methods)
this example is example is showing keyword generation using scikit-learn
the main difference between this and the method used in keyword_gen2 and keyword_gen1 is that scikit-learn has
a built-in functionality to exclude words that either appear too frequently or don't appear frequently enough
Right now, I've yet to figure out part of speech tagging with scikit-learn, so no differentiating verbs and nouns and so on
Also, I haven't been able to work out how to do proper word counts with scikit-learn

before running this script, you need to type this into the project terminal:
     python -m spacy download en_core_web_md 
     this installs the medium sized model used by spacy
     there is also a large and small sized model, but I will only use the medium sized one

type this in the terminal to test the script
     python text_analyzer\example_keyword_gen3.py "text_analyzer\test_data_sentiment_output.csv"
"""


# name of file you want to read and file you want to save to
filepath_input = sys.argv[1]
# filepath_output = sys.argv[2]
# this is just so I can append a suffix to the end of the file name if I want to
# filename = filepath_output.removesuffix('.csv')

comment_data = pd.read_csv(filepath_input)
# this line looks at the comment_data dataframe and removes any row where its corresponding roberta_neg > 0.5
# this means that the comment has been identified to be more than 50% negative
# it then sorts the data in decending order, meaning the largest values (the most negative) will be at the top
sorted_data = comment_data.query('roberta_neg > 0.5').sort_values('roberta_neg', ascending=False).reset_index(drop=True)

spacy_handler = SpacyHandler()
# this line goes through the comment data list and finds all adjectives and verbs
# the repeat=True indicates that we want repeated words to be added to the list
# this is so we can track frequency of a word's occurrences
# e.g. the list output list will look something like this ["bad", "good", "stupid", "go", "sell", "bad", "bad", ....]
# in this example we'll only be pulling adjectives
word_list = spacy_handler.create_list_of_words(to_include="",
                                               text_list=sorted_data['snippet.textOrginal'],
                                               repeat=True)
# we then take the 100 most frequent words and put them in a list
freq_list = keyword_gen_using_tokens(word_list, 10)
# we use that list to create a dataframe which we use to save to a new file
# you don't need to do this, this is just so you can see what words have been pulled
freq_df = pd.DataFrame(freq_list)

keyword_count_vec = keyword_gen_using_sentences(sorted_data['snippet.textOrginal'], 10)

for keyword_dict in freq_list:
    print(keyword_dict)
print("\n\n")
for keyword_dict in keyword_count_vec:
    print(keyword_dict)
