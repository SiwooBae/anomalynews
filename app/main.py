import torch
def main():
    print("Hello from anomalynews!")
    x = torch.rand(5, 3)
    print("PyTorch version:", torch.__version__)
    print("MPS available (Apple Silicon):", torch.backends.mps.is_available())
    x = x.to(torch.device("mps"))
    print(x)


if __name__ == "__main__":
    main()
