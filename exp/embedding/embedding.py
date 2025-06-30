import torch
from ..dataset.gpt_dataset import create_dataloader_v1

def main():
    torch.manual_seed(123)
    vocab_size = 50257
    output_dim = 256
    # input_ids = torch.tensor([2, 3, 5, 1])
    # embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # print(embedding_layer.weight)
    # print(embedding_layer(torch.tensor([3])))
    # print(embedding_layer(input_ids))
    
    with open("data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length,
        shuffle=False
    )
    
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

if __name__=="__main__":
    main()