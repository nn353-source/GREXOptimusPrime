import logging

def createLogger():
    outlogger = logging.getLogger(__name__)
    outlogger.setLevel(logging.DEBUG) # Logger will process messages from DEBUG up

    file_handler = logging.FileHandler("log.log")
    file_handler.setLevel(logging.DEBUG) # Handler will write messages from DEBUG up

    outlogger.addHandler(file_handler)
    outlogger.debug("test")
    return outlogger