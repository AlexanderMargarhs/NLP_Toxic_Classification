# This script simply outputs some plots (as PNGs) of summary statistics of the dataset.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Load pre-trained word vectors
    EMBEDDING = 'Data_GR/glove.840B.300d.txt'

    # Save training and testing Data
    TRAIN_DATA = 'Data_GR/train.csv'
    TEST_DATA = 'Data_GR/test.csv'
    SAMPLE_SUB = 'Data_GR/labels.csv'

    # Load Data into pandas
    train = pd.read_csv(TRAIN_DATA)
    test = pd.read_csv(TEST_DATA)
    submission = pd.read_csv(SAMPLE_SUB)

    list_train = train["tweet"].fillna("_na_").values
    list_test = test["tweet"].fillna("_na_").values

    row_sums = train.iloc[:, 2:].sum(axis=1)
    train['Not Offensive'] = (row_sums == 0)

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

    ax.set(xticklabels=["Offensive", "Not Offensive"])
    plt.savefig('Images/Imbalance.png')
