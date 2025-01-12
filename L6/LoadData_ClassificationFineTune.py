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


download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
print(df.head())

print(df["Label"].value_counts())