import os 
import sys 
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from L7.LoadData import data
from L7.UtilFunctions import format_input

model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"
print(model_input + desired_response)

train_len = int(len(data) * 0.85) 
test_len = int(len(data) * 0.1) 
val_len = train_len - test_len

train_data = data[:train_len]
test_data = data[train_len : train_len + test_len]
val_data = data[train_len + test_len :]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))