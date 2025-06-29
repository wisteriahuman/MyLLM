import re

with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30] if len(preprocessed) >= 30 else None)

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
# print(vocab_size)

vocab = {token : idx for idx, token in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break