with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()

print("Total number of character:", len(raw_text))
print(raw_text[:99])
