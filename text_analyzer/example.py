import feature_sentiment_keyword
import pandas as pd

df = pd.read_csv("shoeData.csv")

# creates a list of sentiments which is then used to create a new dataframe
sentiment_df = pd.DataFrame(feature_sentiment_keyword.generate_sentiment(comment_list=df["Review Description"]))
# add that new dataframe to the original dataframe by columns
df = pd.concat([df, sentiment_df], axis=1)
print(df.head())

# generate keywords
keyword_dict = feature_sentiment_keyword.generate_keywords(df["Review Description"], 25)
print(keyword_dict)
