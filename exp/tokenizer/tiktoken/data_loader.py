import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("../../../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
enc_text = tokenizer.encode(raw_text)
enc_sample = enc_text[50:]
print(len(enc_text))

context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size+1]
# print(f"x: {x}")
# print(f"y:      {y}")

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    # print(context, "---->", desired)
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    