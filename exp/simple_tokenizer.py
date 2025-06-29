import re


class SimpleTokenizerV1:
    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}
    
    
    def encode(self, text: str) -> list:
        """入力テキストをトークンIDのベクトルに変換

        Args:
            text (str): 入力テキスト

        Returns:
            list: トークンIDのベクトル
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids: int) -> str:
        """トークンをテキストに変換する

        Args:
            ids (int): トークンIDのベクトル

        Returns:
            str: 変換後のテキスト
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab: dict):
        self.str_to_int = vocab
        self.int_to_str = { i : s for s, i in vocab.items()}
    
    def encode(self, text: str) -> list:
        """入力テキストをトークンIDのベクトルに変換

        Args:
            text (str): 入力テキスト

        Returns:
            list: トークンIDのベクトル
        """
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids: list) -> str:
        """トークンをテキストに変換する

        Args:
            ids (int): トークンIDのベクトル

        Returns:
            str: 変換後のテキスト
        """
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text


def main():
    with open("../data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print(len(preprocessed))
    # print(preprocessed[:30] if len(preprocessed) >= 30 else None)

    # all_words = sorted(set(preprocessed))
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    # vocab_size = len(all_words)
    # print(vocab_size)

    # vocab = {token : idx for idx, token in enumerate(all_words)}
    vocab = {token : idx for idx, token in enumerate(all_tokens)}
    
    # for i, item in enumerate(vocab.items()):
    #     print(item)
    #     if i >= 50:
    #         break
    tokenizer = SimpleTokenizerV2(vocab)
    # text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))

if __name__=="__main__":
    main()