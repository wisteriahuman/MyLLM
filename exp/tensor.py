import torch


def main():
    tensor0d = torch.tensor(1)
    tensor1d = torch.tensor([1, 2, 3])
    tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    # tensor print dtype and change dtype
    # print(tensor1d)
    # print(tensor1d.dtype)
    # floatvec = tensor1d.to(torch.float32)
    # print(floatvec.dtype)

    # tensor shape and reshape
    # print(tensor2d.shape)
    # print(tensor2d)
    # print(tensor2d.reshape(3, 2))
    # print(tensor2d.view(3, 2))
    
    # tensor T
    # print(tensor2d.T)
    
    # tensor matmul
    # print(tensor2d)
    # print(tensor2d.T)
    # print(tensor2d.matmul(tensor2d.T))
    # print(tensor2d @ tensor2d.T)


if __name__ == "__main__":
    main()
