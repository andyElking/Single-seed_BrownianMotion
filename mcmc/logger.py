from abc import abstractmethod
from typing import Optional


class AbstractLogger:
    def __init__(
        self, log_filename: Optional[str] = None, print_to_console: bool = True
    ):
        self.log_filename = log_filename
        self.print_to_console = print_to_console

    @abstractmethod
    def make_log_string(self, method_name: str, method_dict: dict) -> str:
        raise NotImplementedError

    def print_log(self, log_str: str):
        if self.log_filename is not None:
            with open(self.log_filename, "a") as f:
                f.write(log_str + "\n")
        if self.print_to_console:
            print(log_str)

    def start_log(self, timestamp: str):
        if self.log_filename is not None:
            with open(self.log_filename, "w") as f:
                f.write(f"Results for {timestamp}\n")

    def start_model_section(self, model_name: str):
        self.print_log(f"\n======= {model_name} =======\n")

    def log_method(self, method_name: str, method_dict: dict):
        log_str = self.make_log_string(method_name, method_dict)
        self.print_log(log_str)


class GenericLogger(AbstractLogger):
    def make_log_string(self, method_name: str, method_dict: dict) -> str:
        log_str = f"{method_name}:\n"
        for key, value in method_dict.items():
            # if value is numeric, print it with 4 decimal places
            if isinstance(value, (int, float)):
                log_str += f"{key}: {value:.4f}   "
            else:
                log_str += f"{key}: {value}   "
        return log_str
