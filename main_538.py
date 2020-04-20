from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
