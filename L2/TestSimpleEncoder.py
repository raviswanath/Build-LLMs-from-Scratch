import urllib.request
import re
from Tokenize import SimpleTokenizer

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# processes and removes periods, punctuations, double dashes, question marks etc. 
# does NOT strings like "doesn't"
v2 = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
v2 = [item.strip() for item in v2 if item.strip()]
all_tokens = sorted(set(v2))
# add unk and endofline
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# this is the vocabulary for traingin/testing
vocab = {word: num for num, word in enumerate(all_tokens)}

# initiate tokenizer object
tokenize = SimpleTokenizer(vocab=vocab)

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

print("The encoded text is: ")
print(tokenize.encode(text))

print("\n")
print("The decoded text is: ")
print(tokenize.decode(tokenize.encode(text)))