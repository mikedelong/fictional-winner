import datetime
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import copysign
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import binomial

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

    democrat = 'Biden'
    republican = 'Trump'
    grade_filter = {}
    electoral_college_df, review_2016_df, filtered_df, state_abbreviations = get_data(democrat=democrat,
                                                                                      grade_to_filter=grade_filter,
                                                                                      republican=republican, )
    cutoff_date = pd.Timestamp(datetime.datetime.today())
    realizations = list()
    realization_count = 10000
    count_democrat = 0
    count_republican = 0
    democrat_realizations = list()
    median_results = list()
    done = False
    for index, realization in enumerate(range(realization_count)):
        if not done:
            realization_democrat, realization_republican = \
                get_realization(arg_cutoff_date=cutoff_date, arg_democrat=democrat, arg_df=filtered_df.copy(deep=True, ),
                                arg_republican=republican, electoral_df=electoral_college_df,
                                historical_df=review_2016_df, )
            count_democrat += 1 if realization_democrat > realization_republican else 0
            count_republican += 1 if realization_democrat < realization_republican else 0
            format_string = '{} {}: {} {}: {} {}: {} {}: {} ratio: {:5.4f} mean: {:5.1f} median: {} streak: {}'
            democrat_realizations = [item[0] for item in realizations]
            if len(democrat_realizations):
                median_result = int(np.median(np.array(democrat_realizations)), )
                if median_result not in median_results:
                    median_results = list()
                median_results.append(median_result)
                logger.info(format_string.format(index, democrat, realization_democrat, republican, realization_republican,
                                                 democrat, count_democrat, republican, count_republican,
                                                 count_democrat / (count_democrat + count_republican),
                                                 np.array(democrat_realizations).mean(), median_result,
                                                 len(median_results), ))
            realizations.append((realization_democrat, realization_republican,))
            done = len(median_results) > 200
    bin_count = max(democrat_realizations) - min(democrat_realizations) + 1
    democrat_win_realizations = [item for item in democrat_realizations if item >= 270]
    democrat_lose_realizations = [item for item in democrat_realizations if item < 270]
    logger.info('{} simulated wins: {} out of {} realizations'.format(democrat, len(democrat_win_realizations),
                                                                      len(democrat_realizations),
                                                                      len(democrat_win_realizations) / len(
                                                                          democrat_realizations), ))
    format_string = '{} mean outcome: {:5.2f} median outcome: {:.0f}'
    realization_mean = np.array(democrat_realizations).mean()
    realization_median = np.median(np.array(democrat_realizations))
    logger.info(format_string.format(democrat, realization_mean, realization_median, ), )
    plt.hist(x=democrat_win_realizations, bins=bin_count, color='blue', )
    plt.hist(x=democrat_lose_realizations, bins=bin_count, color='red', )
    plt.savefig('./{}-histogram.png'.format(democrat.lower(), ), )

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
