from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import numpy as np


def get_result(arg_probability):
    return np.random.binomial(n=1, p=arg_probability)


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')
    probability = 0.66
    count = 10000
    result = [get_result(probability) for _ in range(count)]
    logger.info(sum(result)/count)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
