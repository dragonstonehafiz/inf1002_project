import feature_sentiment_keyword
import pandas as pd

filename = "shoeData.csv"
# filename = "test_data_original.csv"
output_file = f"{filename.removesuffix('.csv')}"
# data_col_name = "snippet.textOrginal"
data_col_name = "Review Description"

df = pd.read_csv(filename)

# creates a list of sentiments which is then used to create a new dataframe
sentiment_df = pd.DataFrame(feature_sentiment_keyword.generate_sentiment(comment_list=df[data_col_name]))
# add that new dataframe to the original dataframe by columns
df = pd.concat([df, sentiment_df], axis=1)
df.to_csv(f"{output_file}Output.csv", index=False)
print(df.head())

# generate keywords
keyword_dict = feature_sentiment_keyword.generate_keywords(df[data_col_name], 25)
print(keyword_dict)
