from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import pandas as pd
import json
import numpy as np

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    with open(file='./electoral_college.json', mode='r', ) as electoral_college_fp:
        electoral_college = json.load(fp=electoral_college_fp)

    electoral_college_df = pd.DataFrame.from_dict({'state': list(electoral_college.keys()),
                                                   'votes': list(electoral_college.values())})

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

    # todo note this isn't right either
    review_2016_df = pd.read_csv('./world-population-review.csv')
    logger.info(list(review_2016_df))
    logger.info('2016 result (WPR) : DEM: {} GOP: {}'.format(review_2016_df['electoralDem'].sum(),
                                                             review_2016_df['electoralRep'].sum()))
    # patch up what DC is called here
    review_2016_df['State'] = review_2016_df['State'].replace(to_replace='Washington DC',
                                                              value='District of Columbia', )

    # patch up 2016 Main Congressional District votes
    review_2016_df = review_2016_df.append(
        {'State': 'Maine CD-1', 'votesDem': 212774, 'percD': 53.96, 'votesRep': 154384, 'percR': 39.15,
         'electoralDem': 1, 'electoralRep': 0, 'Pop': 0, }, ignore_index=True)
    # first cut down the data to just the columns we want
    df = df[['question_id', 'state', 'end_date', 'answer', 'pct']]
    df = df[df.answer.isin({'Biden', 'Trump'})]
    df['question_id'] = df['question_id'].astype(int)

    a2_df = df[df.answer.isin({'Biden', 'Trump'})].groupby('question_id').filter(lambda x: len(x) == 2)
    polling = {}
    for state in sorted(a2_df.state.unique()):
        polling[state] = {}
        this_df = a2_df[a2_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        for candidate in ['Biden', 'Trump']:
            polling[state][candidate] = this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]

    biden_votes, trump_votes = 0, 0
    for state in electoral_college_df.state.unique():
        if state in polling.keys():
            poll = polling[state]
            votes = electoral_college_df[electoral_college_df.state == state].votes.values[0]
            if poll['Biden'] > poll['Trump']:
                biden_votes += votes
            elif poll['Biden'] < poll['Trump']:
                trump_votes += votes
            logger.info('state: {} polling margin: {:5.1f} pct'.format(state, abs(poll['Biden'] - poll['Trump'])))
        elif state in review_2016_df.State.unique():
            biden_votes += review_2016_df[review_2016_df.State == state].electoralDem.values[0]
            trump_votes += review_2016_df[review_2016_df.State == state].electoralRep.values[0]
        else:
            logger.warning('missing state: {}'.format(state))
        logger.info('state: {} Biden: {} Trump: {} total: {} remaining: {}'.format(state, biden_votes, trump_votes,
                                                                                   biden_votes + trump_votes,
                                                                                   538 - biden_votes - trump_votes))

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
