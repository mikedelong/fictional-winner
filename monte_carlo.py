from datetime import datetime
from json import load
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import copysign
from pathlib import Path
from time import time

from matplotlib.pyplot import clf
from matplotlib.pyplot import hist
from matplotlib.pyplot import savefig
from matplotlib.pyplot import scatter
from matplotlib.pyplot import style
from matplotlib.pyplot import tight_layout
from matplotlib.pyplot import xticks
from numpy import array
from numpy import median
from numpy.random import binomial
from pandas import DataFrame
from pandas import Timestamp
from pandas import to_datetime
from pandas import read_csv

from get_data import get_data


def get_realization(arg_df, arg_cutoff_date, electoral_df, historical_df, arg_democrat, arg_republican, arg_margin,
                    arg_logger):
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
            if abs(democrat_pct - republican_pct) < arg_margin:
                simulated_democrat_result = binomial(n=1, p=democrat_pct / (democrat_pct + republican_pct))
            else:
                simulated_democrat_result = int((1 + copysign(1, democrat_pct - republican_pct)) / 2)
            result_democrat_votes += votes * simulated_democrat_result
            result_republican_votes += votes * (1 - simulated_democrat_result)
        elif state in review_unique:
            result_democrat_votes += historical_df[historical_df.State == state].electoralDem.values[0]
            result_republican_votes += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            arg_logger.warning('missing state: {}'.format(state), )
    return int(result_democrat_votes), int(result_republican_votes)


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )

    with open(file='./settings.json', mode='r', ) as settings_fp:
        settings = load(fp=settings_fp, )
        logger.info('settings: {}'.format(settings))

    date_range = settings['date_range'] if 'date_range' in settings.keys() else 'one'
    if 'date_range' not in settings.keys():
        logger.warning('date range not in settings; using default [{}]'.format(date_range, ), )
    else:
        logger.info('date range: [{}]'.format(date_range, ), )

    date_ranges = settings['date_ranges'] if 'date_ranges' in settings.keys() else list()
    if 'date_ranges' not in settings.keys():
        logger.warning('date ranges not in settings; quitting.', )
        quit(code=1, )
    else:
        logger.info('date ranges: [{}]'.format(date_range, ), )
    # check the date range against the possible choices
    if date_range in date_ranges:
        logger.info('date range {} is in possible date ranges {}'.format(date_range, date_ranges, ))
    else:
        logger.warning('date range {} not in possible date ranges {}; quitting.'.format(date_range, date_ranges, ))
        quit(code=2, )

    democrat = settings['democrat'] if 'democrat' in settings.keys() else None
    if democrat is None:
        logger.warning('parameter democrat is missing from settings. Quitting.')
        quit(code=3, )

    early_exit_limit = settings['early_exit_limit'] if 'early_exit_limit' in settings.keys() else 100
    if 'early_exit_limit' not in settings.keys():
        logger.warning('early exit limit not in settings; using default value {}'.format(early_exit_limit, ))
    else:
        logger.info('early exit limit: {}'.format(early_exit_limit, ))

    grade_filter = settings['grade_filter'] if 'grade_filter' in settings.keys() else list()
    grade_filter = set(grade_filter)
    if len(grade_filter) == 0:
        logger.warning('grade filter is empty; using all polls')

    output_folder = settings['monte_carlo_output_folder'] if 'monte_carlo_output_folder' in settings.keys() else None
    if 'monte_carlo_output_folder' in settings.keys():
        logger.info('Monte Carlo output folder: {}'.format(output_folder))
    else:
        logger.warning('parameter output folder is missing from settings. Quitting.')
        quit(code=4, )
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    realization_count = settings['realization_count'] if 'realization_count' in settings.keys() else 1000
    if 'realization_count' in settings.keys():
        logger.info('realization count: {}'.format(realization_count))
    else:
        logger.warning('using default realization count: {}'.format(realization_count))

    realization_margin = settings['realization_margin'] if 'realization_margin' in settings.keys() else 10.0
    if 'realization_margin' not in settings.keys():
        logger.warning('realization margin not in settings; using default value {}'.format(realization_margin))
    else:
        logger.info('realization margin: {}'.format(realization_margin))

    republican = settings['republican'] if 'republican' in settings.keys() else None
    if republican is None:
        logger.warning('parameter republican is missing from settings. Quitting.')
        quit(code=5, )

    use_historical = settings['use_historical'] if 'use_historical' in settings.keys() else False

    electoral_college_df, review_2016_df, filtered_df, state_abbreviations = get_data(democrat=democrat,
                                                                                      grade_to_filter=grade_filter,
                                                                                      republican=republican, )
    median_csv = './monte_carlo.csv'
    if use_historical:
        already_df = read_csv(filepath_or_buffer=median_csv)
        logger.info('loaded {} rows from {}'.format(len(already_df), median_csv))
    else:
        already_df = None

    historical_dates = list()
    cutoff_dates = list()
    if date_range == 'one':
        cutoff_dates = [Timestamp(datetime.today())]
    elif date_range == 'all':
        cutoff_dates = [Timestamp(item) for item in sorted(filtered_df['end_date'].unique())]
    elif date_range == '2020-01-01':
        cutoff_dates = [Timestamp(item) for item in sorted(filtered_df['end_date'].unique()) if
                        to_datetime(item) > datetime.strptime('2019-12-31 23:59:59', '%Y-%m-%d %H:%M:%S', )]
    elif date_range == 'update':
        historical_dates = sorted(already_df['date'].values)
        historical_dates = [datetime.strptime(item, '%Y-%m-%d', ).date() for item in historical_dates]
        # drop off the last five dates somewhat arbitrarily
        historical_dates = historical_dates[:-10]
        cutoff_dates = [Timestamp(item) for item in sorted(filtered_df['end_date'].unique()) if
                        Timestamp(item).date() not in historical_dates]
    else:
        logger.warning('unexpected date range [{}]; quitting.'.format(date_range))
        quit(code=5, )
    # we need to create/re-create the median map
    median_map = dict()
    if date_range == 'update':
        for index, row in already_df.iterrows():
            date = row['date']
            if datetime.strptime(date, '%Y-%m-%d', ).date() in historical_dates:
                median_map[Timestamp(date)] = int(row['median'])

    instance_format = '{} {} {}: {} {}: {} {}: {} {}: {} ratio: {:5.4f} mean: {:5.1f} median: {} streak: {}'
    outcome_format = '{} mean outcome: {:5.2f} median outcome: {:.0f}-{:.0f}'
    wins_format = '{} {} simulated wins: {} out of {} realizations'
    rotation = 45
    for cutoff_date in cutoff_dates:
        count_democrat = 0
        count_republican = 0
        democrat_realizations = list()
        done = False
        median_results = list()
        realizations = list()
        for index, realization in enumerate(range(realization_count)):
            if not done:
                realization_democrat, realization_republican = \
                    get_realization(arg_cutoff_date=cutoff_date, arg_democrat=democrat,
                                    arg_df=filtered_df.copy(deep=True, ), arg_logger=logger,
                                    arg_margin=realization_margin, arg_republican=republican,
                                    electoral_df=electoral_college_df, historical_df=review_2016_df, )
                count_democrat += 1 if realization_democrat > realization_republican else 0
                count_republican += 1 if realization_democrat < realization_republican else 0
                democrat_realizations = [item[0] for item in realizations]
                if len(democrat_realizations):
                    median_result = int(median(array(democrat_realizations)), )
                    if median_result not in median_results:
                        median_results = list()
                    median_results.append(median_result)
                    logger.info(
                        instance_format.format(cutoff_date.date(), index, democrat, realization_democrat, republican,
                                               realization_republican, democrat, count_democrat, republican,
                                               count_republican, count_democrat / (count_democrat + count_republican),
                                               array(democrat_realizations).mean(), median_result,
                                               len(median_results), ), )
                realizations.append((realization_democrat, realization_republican,))
                done = len(median_results) > early_exit_limit
        bin_count = max(democrat_realizations) - min(democrat_realizations) + 1
        democrat_win_realizations = [item for item in democrat_realizations if item > 270]
        democrat_lose_realizations = [item for item in democrat_realizations if item < 270]
        logger.info(
            wins_format.format(cutoff_date, democrat, len(democrat_win_realizations), len(democrat_realizations),
                               len(democrat_win_realizations) / len(democrat_realizations), ), )
        realization_mean = array(democrat_realizations).mean()
        realization_median = int(median(array(democrat_realizations)))
        logger.info(outcome_format.format(democrat, realization_mean, realization_median, 538 - realization_median), )
        style.use('fivethirtyeight')
        if len(democrat_win_realizations) == 0:
            range_ = (min(democrat_lose_realizations), max(democrat_lose_realizations))
        elif len(democrat_lose_realizations) == 0:
            range_ = (min(democrat_win_realizations), max(democrat_win_realizations))
        else:
            range_ = (min(democrat_lose_realizations), max(democrat_win_realizations))
        hist(bins=bin_count, color='blue', range=range_, x=democrat_win_realizations, )
        hist(bins=bin_count, color='red', range=range_, x=democrat_lose_realizations, )
        tight_layout()
        output_file = '{}{}-{}-histogram.png'.format(output_folder, democrat.lower(), cutoff_date.date(), )
        logger.info('writing {} histogram to {}'.format(cutoff_date.date(), output_file, ), )
        savefig(output_file, )
        clf()
        median_map[cutoff_date] = realization_median
        if len(median_map) > 1:
            xs = sorted(list(median_map.keys()))
            ys = [median_map[key] for key in xs]
            colors = ['b' if y > 269 else 'r' for y in ys]
            scatter(c=colors, x=xs, y=ys, )
            xticks(rotation=rotation, )
            tight_layout()
            median_file_name = '{}median.png'.format(output_folder, )
            logger.info('writing updated median data to {}'.format(median_file_name, ), )
            savefig(median_file_name, )
            clf()
            margin_ys = [2 * y - 538 for y in ys]
            margin_colors = ['b' if y > 0 else 'r' for y in margin_ys]
            scatter(c=margin_colors, x=xs, y=margin_ys, )
            xticks(rotation=rotation, )
            tight_layout()
            margin_file_name = '{}margin.png'.format(output_folder, )
            logger.info('writing updated margin data to {}'.format(margin_file_name, ), )
            savefig(margin_file_name, )
            clf()
            logger.info('{} totals: {}'.format(democrat, ys, ), )
            logger.info('margin values: {}'.format(margin_ys, ), )
            median_df = DataFrame(
                data={'date': list(median_map.keys()), 'median': list(median_map.values()), 'margin': margin_ys})
            logger.info('writing current state data to {}'.format(median_csv))
            median_df.to_csv(index=True, header=True, path_or_buf=median_csv, )

    if len(median_map) > 1:
        logger.info(median_map)
    logger.info('total time: {:5.2f}s'.format(time() - time_start))
