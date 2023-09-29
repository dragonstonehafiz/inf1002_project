import pandas as pd
import sys
from analyzer_spacy import SpacyHandler
from analyzer_keyword_gen import keyword_gen_using_tokens

""" 
PLEASE READ
before running this script, you need to type this into the project terminal:
     python -m spacy download en_core_web_md 
     this installs the medium sized model that used by spacy
     there is also a large and small sized model, but I will only use the medium sized one
    
type this in the terminal to test the script
    python text_analyzer\example_keyword_gen1.py "text_analyzer\test_data_original.csv" "text_analyzer\test_data_keyword_gen1.csv" 
"""

# name of file you want to read and file you want to save to
filepath_input = sys.argv[1]
filepath_output = sys.argv[2]
# this is just so I can append a suffix to the end of the file name if I want to
filename = filepath_output.removesuffix('.csv')
# for this test we will only be looking at the first 1500 entries
comment_data = pd.read_csv(filepath_input).head(1500)

spacy_handler = SpacyHandler()
# this line goes through the comment data list and finds all adjectives and verbs
# the repeat=True indicates that we want repeated words to be added to the list
# this is so we can track frequency of a word's occurrences
# e.g. the list output list will look something like this ["bad", "good", "stupid", "go", "sell", "bad", "bad", ....]
word_list = spacy_handler.create_list_of_words(to_include="ADJ,VERB",
                                               text_list=comment_data['snippet.textOrginal'],
                                               repeat=True)
# we then take the 100 most frequent words and put them in a list
freq_list = keyword_gen_using_tokens(word_list, 100)
# we use that list to create a dataframe which we use to save to a new file
# you don't need to do this, this is just so you can see what words have been pulled
freq_df = pd.DataFrame(freq_list)
freq_df.to_csv(filepath_output, index=False)

