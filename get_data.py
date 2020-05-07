import json
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import pandas as pd
from pandas.plotting import register_matplotlib_converters


def get_data(democrat, republican):
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )
    register_matplotlib_converters()

    with open(file='./electoral_college.json', mode='r', ) as electoral_college_fp:
        electoral_college = json.load(fp=electoral_college_fp, )

    electoral_college_df = pd.DataFrame.from_dict({'state': list(electoral_college.keys()),
                                                   'votes': list(electoral_college.values())})

    logger.info('Electoral College: {} total votes.'.format(electoral_college_df['votes'].sum()), )

    with open(file='./state_abbreviations.json', mode='r', ) as abbreviation_fp:
        state_abbreviations = json.load(fp=abbreviation_fp, )

    url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'
    df = pd.read_csv(url, parse_dates=['end_date'], )
    logger.info(list(df))
    # remove null states (these are nationwide)
    df = df[~df.state.isnull()]
    # remove unrated pollsters
    df = df[~df.fte_grade.isnull()]
    logger.info(sorted(df.state.unique()))
    logger.info(df.state.nunique())
    polls_no_electoral_college = [state for state in sorted(df.state.unique()) if state not in electoral_college.keys()]
    electoral_college_no_polls = [state for state in sorted(electoral_college.keys()) if state not in df.state.values]
    if len(polls_no_electoral_college):
        for item in polls_no_electoral_college:
            logger.warning('no Electoral College data for {}'.format(item), )
    if len(electoral_college_no_polls):
        for item in electoral_college_no_polls:
            logger.warning('no polls for {}'.format(item), )

    review_2016_df = pd.read_csv('./world-population-review.csv', )
    logger.info(list(review_2016_df), )
    # patch up what DC is called here
    review_2016_df['State'] = review_2016_df['State'].replace(to_replace='Washington DC',
                                                              value='District of Columbia', )
    # patch up 2016 Maine Congressional District votes
    review_2016_df = review_2016_df.append(ignore_index=True,
                                           other={'State': 'Maine CD-1', 'votesDem': 212774, 'percD': 53.96,
                                                  'votesRep': 154384, 'percR': 39.15, 'electoralDem': 1,
                                                  'electoralRep': 0, 'Pop': 394329, }, )
    review_2016_df = review_2016_df.append(ignore_index=True,
                                           other={'State': 'Maine CD-2', 'votesDem': 144817, 'percD': 40.98,
                                                  'votesRep': 181177, 'percR': 51.26, 'electoralDem': 0,
                                                  'electoralRep': 1, 'Pop': 353416, }, )
    # patch up 2016 Nebraska Congressional District votes
    review_2016_df = review_2016_df.append(ignore_index=True,
                                           other={'State': 'Nebraska CD-1', 'votesDem': 100126, 'percD': 35.46,
                                                  'votesRep': 158626, 'percR': 56.18, 'electoralDem': 0,
                                                  'electoralRep': 1, 'Pop': 282338, }, )
    review_2016_df = review_2016_df.append(ignore_index=True,
                                           other={'State': 'Nebraska CD-2', 'votesDem': 131030, 'percD': 44.92,
                                                  'votesRep': 137564, 'percR': 47.16, 'electoralDem': 0,
                                                  'electoralRep': 1, 'Pop': 291680, }, )
    review_2016_df = review_2016_df.append(ignore_index=True,
                                           other={'State': 'Nebraska CD-3', 'votesDem': 53290, 'percD': 19.73,
                                                  'votesRep': 199657, 'percR': 73.92, 'electoralDem': 0,
                                                  'electoralRep': 1, 'Pop': 270109, }, )
    # fix some errors in our Electoral College data
    review_2016_df.loc[review_2016_df.State == 'Hawaii', 'electoralDem'] = 4
    review_2016_df.loc[review_2016_df.State == 'Nebraska', 'electoralRep'] = 2
    review_2016_df.loc[review_2016_df.State == 'Texas', 'electoralRep'] = 38
    review_2016_df.loc[review_2016_df.State == 'Washington', 'electoralDem'] = 12

    dem_2016_total = review_2016_df['electoralDem'].sum()
    rep_2016_total = review_2016_df['electoralRep'].sum()
    total_2016 = dem_2016_total + rep_2016_total
    logger.info('2016 result (WPR) : DEM: {} GOP: {}: total: {} missing: {}'.format(dem_2016_total, rep_2016_total,
                                                                                    total_2016, 538 - total_2016))

    review_2016_df['electoralTotal'] = review_2016_df['electoralDem'] + review_2016_df['electoralRep']
    check_df = review_2016_df[['State', 'electoralTotal']].copy(deep=True).merge(how='inner', left_on='State',
                                                                                 right=electoral_college_df,
                                                                                 right_on='state', ).drop(['state'],
                                                                                                          axis=1, )
    check_df = check_df[check_df.votes != check_df.electoralTotal]

    # first cut down the data to just the columns we want
    df = df[['question_id', 'state', 'end_date', 'answer', 'pct', 'fte_grade', ]]
    df = df[df.answer.isin({democrat, republican, })]
    df['question_id'] = df['question_id'].astype(int)

    a2_df = df[df.answer.isin({democrat, republican, })].groupby('question_id').filter(lambda x: len(x) == 2)
    # filter out low-grade polls (?)
    a2_df = a2_df[a2_df.fte_grade.isin(['A+', 'A', 'A-', 'A/B', 'B', 'B-', 'B/C', 'C', ])]

    logger.info('total time: {:5.2f}s'.format(time() - time_start, ))

    return electoral_college_df, review_2016_df, a2_df, state_abbreviations
