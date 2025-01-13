import urllib.request
import zipfile, os 
from pathlib import Path 
import pandas as pd


url = 'https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip'
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction")
        return 
    
    # download the file 
    with urllib.request.urlopen(url) as response: 
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
    

    # unzip the file 
    with zipfile.ZipFile(zip_path, "r") as zip_ref: 
        zip_ref.extractall(extracted_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved at path {original_file_path}")


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([ham_subset, df[df.Label == "spam"]])
    return balanced_df


def random_split(df, train_frac, val_frac):

    # shuffle the dataset first
    df = df.sample(frac=1).reset_index(drop=True)

    train_end = int(df.shape[0] * train_frac)
    val_end = train_end + int(df.shape[0] * val_frac)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test


download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

print(train_df.shape)
print(validation_df.shape)
print(test_df.shape)