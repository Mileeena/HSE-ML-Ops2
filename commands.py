import subprocess
import sys

import fire


class Commands:
    def download(self, data_dir: str = "./data") -> None:
        from fashion_mnist_classifier.data.download import download_data

        download_data(data_dir)

    def train(self) -> None:
        from fashion_mnist_classifier.training.train import train

        train()

    def dvc_pull(self) -> None:
        print("Забираем данные из DVC...")
        subprocess.run(["dvc", "pull"], check=True)

    def dvc_push(self) -> None:
        print("Отправляем данные в DVC...")
        subprocess.run(["dvc", "push"], check=True)

    def dvc_status(self) -> None:
        subprocess.run(["dvc", "status"], check=True)

    def precommit_install(self) -> None:
        subprocess.run(["pre-commit", "install"], check=True)
        print("Pre-commit hooks install")

    def precommit_run(self, all_files: bool = True) -> None:
        cmd = ["pre-commit", "run"]
        if all_files:
            cmd.append("--all-files")
        subprocess.run(cmd)

    def mlflow_ui(self, port: int = 8080) -> None:
        print(f"MLflow on http://127.0.0.1:{port}")
        subprocess.run(["mlflow", "ui", "--port", str(port)])

    def docker_up(self) -> None:
        print("Start Docker services...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)

    def docker_down(self) -> None:
        print("Stop Docker services...")
        subprocess.run(["docker-compose", "down"], check=True)

    def docker_logs(self, service: str = "mlflow") -> None:
        subprocess.run(["docker-compose", "logs", "-f", service])


def main():
    fire.Fire(Commands)


if __name__ == "__main__":
    sys.exit(main())
