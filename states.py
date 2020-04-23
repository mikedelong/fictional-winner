from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import pandas as pd
import json

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open(file='./electoral_college.json', mode='r', ) as electoral_college_fp:
        electoral_college = json.load(fp=electoral_college_fp)

    url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'

    df = pd.read_csv(url, parse_dates=['end_date'])
    logger.info(list(df))
    df = df[~df.state.isnull()]
    logger.info(sorted(df.state.unique()))
    logger.info(df.state.nunique())
    polls_no_electoral_college = [state for state in sorted(df.state.unique()) if state not in electoral_college.keys()]
    electoral_college_no_polls = [state for state in sorted(electoral_college.keys()) if state not in df.state.values]
    if len(polls_no_electoral_college):
        for item in polls_no_electoral_college:
            logger.warning('no Electoral College data for {}'.format(item))
    if len(electoral_college_no_polls):
        for item in electoral_college_no_polls:
            logger.warning('no polls for {}'.format(item))

    # todo get 2016 data to fill in data for states with no polls
    # todo build a simple model that uses the most recent poll for each state
    state_data_url = 'https://raw.githubusercontent.com/john-guerra/US_Elections_Results/master/US%20presidential%20' \
                     'election%20results%20by%20county.csv'
    results_2016_df = pd.read_csv(state_data_url, usecols=['state_abbr', 'votes_dem', 'votes_gop'])
    logger.info(results_2016_df.shape)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
