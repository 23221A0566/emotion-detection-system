from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "."
)

print("SUCCESS")