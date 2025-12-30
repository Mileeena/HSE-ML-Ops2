from pathlib import Path

from torchvision.datasets import FashionMNIST


def download_data(data_dir: str = "./data") -> None:
    """
    Скачивание Fashion-MNIST из torchvision.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    print(f"Начало скачивания Fashion-MNIST в {data_dir}...")

    FashionMNIST(root=data_dir, train=True, download=True)
    FashionMNIST(root=data_dir, train=False, download=True)

    print("Скачивание завершено")
    print(f"   Train: {data_dir}/FashionMNIST/raw/train-*.gz")
    print(f"   Test:  {data_dir}/FashionMNIST/raw/t10k-*.gz")


if __name__ == "__main__":
    download_data()
