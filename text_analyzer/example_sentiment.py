import pandas as pd
import sys
from analyzer_sentiment import sentiment_analyzer_roberta

"""
PLEASE READ
you may get this error when running the program for the first time:
    ImportError:
    AutoModelForSequenceClassification requires the PyTorch library but it was not found in your environment. Checkout the instructions on the

if that happens, you need to install the PyTorch library which is used by huggingface:
    follow this site's instructions https://pytorch.org/get-started/locally/
    I think it may take 2gb of space, so keep that in mind

to test this program, type this into your project terminal
    python text_analyzer\example_sentiment.py "text_analyzer\test_data_original.csv" "text_analyzer\test_data_sentiment_output.csv"
    
    Data is list of YouTube comments I pulled from this video by Tom Scott: https://www.youtube.com/watch?v=1Jwo5qc78QU
    
    note that during your first time running this file it may take a little longer to start up
    
    also note, you will see something like  "a gwrAc7wvWhI_TAuFS54AaABAg unable to be analyzed.
                                            "ERROR: The expanded size of the tensor (521)...."
    it won't cause a crash, but it means that one comment couldn't be analyzed. Couldn't figure out why, but it should
    not be a problem
"""

# name of file you want to read and file you want to save to
filepath_input = sys.argv[1]
filepath_output = sys.argv[2]
# this is just so I can append a suffix to the end of the file name if I want to
filename = filepath_output.removesuffix('.csv')
# the .head(500) makes it so that comment_data only has the first 500 data entries
# this is done purely because the test_data.csv file has over 10,000 entries.
# If I'm not misremembering, analyzing the whole thing took me a little over 15 minutes
# if data is limited to 2000 analysis should take no longer than five minute
comment_data = pd.read_csv(filepath_input).head(2000)

# this creates the list of dictionaries with this structure:
# [{"id":str,"roberta_neg":float,"roberta_neu":float,"roberta_pos":float}, ....]
# if you want to use your different data:
#   change comment_data['snippet.textOrginal'] to a list of whatever data you want to test
#   comment_data['id'] is only there because I need it to merge the new data with the original dataset
roberta_data_dict = sentiment_analyzer_roberta(comment_data['snippet.textOrginal'], comment_data['id'])
# then create a dataframe to store all the comments and their polarity scores
roberta_data_df = pd.DataFrame(roberta_data_dict)
# merge this dataframe with the original dataframe using at the ['id'] column
sentiment_df_roberta = roberta_data_df.merge(comment_data, how='left')
# Save the dataframe to a csv file
# can be the same file or a different one
sentiment_df_roberta.to_csv(filepath_output, index=False)


