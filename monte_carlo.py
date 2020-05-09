import datetime
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import copysign
from math import trunc
from time import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import binomial
from pandas.plotting import register_matplotlib_converters

from get_data import get_data




def get_realization(arg_df, arg_cutoff_date, electoral_df, historical_df, arg_democrat, arg_republican):
    polling = {}
    arg_df = arg_df[arg_df.end_date <= arg_cutoff_date]
    for state in arg_df.state.unique():
        polling[state] = {}
        this_df = arg_df[arg_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        for candidate in [arg_democrat, arg_republican]:
            polling[state][candidate] = this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]
    result_democrat_votes = 0
    result_republican_votes = 0
    review_unique = historical_df.State.unique()
    for state in electoral_df.state.unique():
        if state in polling.keys():
            poll = polling[state]
            votes = electoral_df[electoral_df.state == state].votes.values[0]
            democrat_pct = poll[arg_democrat]
            republican_pct = poll[arg_republican]
            if abs(democrat_pct - republican_pct) < 15.1:
                simulated_democrat_result = binomial(n=1, p=democrat_pct / (democrat_pct + republican_pct))
            else:
                simulated_democrat_result = int((1 + copysign(1, democrat_pct - republican_pct)) / 2)
            result_democrat_votes += votes * simulated_democrat_result
            result_republican_votes += votes * (1 - simulated_democrat_result)
        elif state in review_unique:
            result_democrat_votes += historical_df[historical_df.State == state].electoralDem.values[0]
            result_republican_votes += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            logger.warning('missing state: {}'.format(state), )
    return result_democrat_votes, result_republican_votes


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
