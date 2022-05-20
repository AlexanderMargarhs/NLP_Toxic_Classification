import pandas as pd

if __name__ == "__main__":
	TRAIN_DATA = 'Data/test.csv'
	SAMPLE_SUB = 'Data/test_labels.csv'

	train = pd.read_csv(TRAIN_DATA)
	submission = pd.read_csv(SAMPLE_SUB)

	df_merge_col = pd.merge(train, submission, on='id')

	df_merge_col.drop("comment_text", axis=1, inplace=True)

	df_merge_col.to_csv('Data/test_labels.csv', index=False)
