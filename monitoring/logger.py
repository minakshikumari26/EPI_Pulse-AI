from src.utils.logger import get_logger


def log_message(message):
    get_logger("epipulse.monitoring").info(message)


if __name__ == "__main__":
    log_message("Monitoring logger initialized")

