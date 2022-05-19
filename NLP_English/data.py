# This script simply outputs some plots (as PNGs) of summary statistics of the dataset.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load pre-trained word vectors
    EMBEDDING = 'Data/glove.840B.300d.txt'

    # Save training and testing Data
    TRAIN_DATA = 'Data/train.csv'
    TEST_DATA = 'Data/test.csv'
    SAMPLE_SUB = 'Data/sample_submission.csv'

    # Load Data into pandas
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    submission = pd.read_csv(SAMPLE_SUB)

    list_train = train["comment_text"].fillna("_na_").values
    list_test = test["comment_text"].fillna("_na_").values

    # Label comments with no tag as "clean"
    row_sums = train.iloc[:, 2:].sum(axis=1)
    train['clean'] = (row_sums == 0)

    # Plot class imbalance
    x = train.iloc[:, 2:].sum()
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x.index, x.values)
    plt.title("Number of Sentences Per Class in Training Data")
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Class ', fontsize=12)
    rectangles = ax.patches
    labels = x.values
    for rect, label in zip(rectangles, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

    plt.savefig('Images/Imbalance.png')

    # Remove the clean column from the dataframe
    no_clean = train.iloc[:, 2:-1]

    col_1 = "toxic"
    corr = []
    for other_col in no_clean.columns[1:]:
        ct = pd.crosstab(no_clean[col_1], no_clean[other_col])
        corr.append(ct)