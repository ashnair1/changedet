import logging
import sys
from pathlib import Path

from termcolor import colored


class _ColorFormatter(logging.Formatter):
    """
    Color Logging Formatter

    Refer: https://github.com/tensorpack/dataflow/blob/master/dataflow/utils/logger.py
    """

    def format(self, record):
        date = colored("[%(asctime)s]:%(name)s:%(module)s:%(lineno)d:%(levelname)s:", "green")
        msg = "%(message)s"
        if record.levelno == logging.WARNING:
            fmt = date + " " + colored("WRN", "red", attrs=["blink"]) + " " + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + " " + colored("ERR", "red", attrs=["blink", "underline"]) + " " + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + " " + colored("DBG", "yellow", attrs=["blink"]) + " " + msg
        else:
            fmt = date + " " + msg
        if hasattr(self, "_style"):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_ColorFormatter, self).format(record)


def init_logger(name="logger", output=None):
    """
    Initialise changedet logger

    Args:
        name (str, optional): Name of this logger. Defaults to "logger".
        output (str, optional): Path to folder/file to write logs. If None, logs are not written
    """
    logger = logging.getLogger(name=name)
    logger.setLevel(logging.DEBUG)

    # Output logs to terminal
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(_ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(streamhandler)

    # Output logs to file
    if output:
        output = Path(output)
        if output.suffix in [".txt", ".log"]:
            logfile = output
        else:
            logfile = output / "log.txt"
        Path.mkdir(output.parent)

        filehandler = logging.FileHandler(logfile)
        filehandler.setLevel(logging.DEBUG)
        filehandler.setFormatter(_ColorFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(filehandler)
    return logger
