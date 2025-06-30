# from importlib.metadata import version
import tiktoken

# print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
# text = (
#     "Hello, do you lik tea? <|endoftext|> IN the sunlit terraces"
#     "Of someunknownPlace."
# )
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)

"""practice2-1"""
text = "Akwirw ier"
print(text)
integers = tokenizer.encode(text)
print(integers)
strings = [tokenizer.decode([integer]) for integer in integers]
print(strings)

