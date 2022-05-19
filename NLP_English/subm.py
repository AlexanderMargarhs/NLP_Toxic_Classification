import pandas as pd

if __name__ == "__main__":
	lstm_1_df = pd.read_csv('Predictions/submission15.csv')
	lstm_2_df = pd.read_csv('Predictions/submission16.csv')
	gru_1_df = pd.read_csv('Predictions/submission17.csv')
	gru_2_df = pd.read_csv('Predictions/submission18.csv')

	ensemble = lstm_1_df.copy()
	cols = ensemble.columns
	cols = cols.tolist()
	cols.remove('id')
	for i in cols:
		ensemble[i] = (lstm_1_df[i] + lstm_2_df[i] + gru_1_df[i] + gru_2_df[i]) / 4

	ensemble.to_csv('Predictions/ensemble_embeds.csv', index=False)
