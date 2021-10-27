import logging


def load_logger(log_file):
    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    fh = logging.FileHandler(log_file, 'a', 'utf8')
    ch.setLevel(logging.DEBUG)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# print(logger.info("hello"))
