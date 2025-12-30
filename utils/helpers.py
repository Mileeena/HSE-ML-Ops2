import subprocess
from pathlib import Path


def ensure_data_available(data_dir: str = "./data") -> None:
    data_path = Path(data_dir)

    if not data_path.exists() or not any(data_path.iterdir()):
        print("Данные не найдены локально, забираем из DVC...")
        try:
            subprocess.run(["dvc", "pull"], check=True)
            print("Данные успешно получены из DVC")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при получении данных из DVC: {e}")
            print("Пытаемся скачать данные...")
            from fashion_mnist_classifier.data.download import download_data

            download_data(data_dir)
            print("Данные успешно скачаны")


def get_git_commit_id() -> str:
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"])
        return commit_id.decode("ascii").strip()
    except Exception:
        return "unknown"
