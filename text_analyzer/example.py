import feature_sentiment_keyword
import pandas as pd

filename = "overallShoeData.csv"
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

# df = pd.read_csv(f"{output_file}Output.csv")

average_list = feature_sentiment_keyword.generate_average_polarity_scores(name_list=df["Shoe Name"],
                                                                          neg_list=df["roberta_neg"],
                                                                          neu_list=df["roberta_neu"],
                                                                          pos_list=df["roberta_pos"])

print(average_list)