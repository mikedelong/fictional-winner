import datetime
import json
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import trunc
from time import time

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import matplotlib.dates as mdates


def get_results(arg_df, arg_cutoff_date, verbose):
    polling = {}
    arg_df = arg_df[arg_df.end_date <= arg_cutoff_date]
    for state in sorted(arg_df.state.unique()):
        polling[state] = {}
        this_df = arg_df[arg_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        for candidate in ['Biden', 'Trump']:
            polling[state][candidate] = this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]
    result_biden_votes, result_trump_votes = 0, 0
    result_ranked = list()
    for state in electoral_college_df.state.unique():
        if state in polling.keys():
            poll = polling[state]
            votes = electoral_college_df[electoral_college_df.state == state].votes.values[0]
            if poll['Biden'] > poll['Trump']:
                result_biden_votes += votes
            elif poll['Biden'] < poll['Trump']:
                result_trump_votes += votes
            if poll['Biden'] - poll['Trump'] > 0 and verbose:
                logger.info('state: {} polling margin: D+{:3.1f} pct'.format(state, abs(poll['Biden'] - poll['Trump'])))
            elif poll['Biden'] - poll['Trump'] < 0 and verbose:
                logger.info('state: {} polling margin: R+{:3.1f} pct'.format(state, abs(poll['Biden'] - poll['Trump'])))
            else:
                if verbose:
                    logger.info('state: {} tied.'.format(state))
            result_ranked.append((state, poll['Biden'] - poll['Trump']))
        elif state in review_2016_df.State.unique():
            result_biden_votes += review_2016_df[review_2016_df.State == state].electoralDem.values[0]
            result_trump_votes += review_2016_df[review_2016_df.State == state].electoralRep.values[0]
        else:
            if verbose:
                logger.warning('missing state: {}'.format(state))
    return result_biden_votes, result_trump_votes, result_ranked


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')
    register_matplotlib_converters()

    with open(file='./electoral_college.json', mode='r', ) as electoral_college_fp:
        electoral_college = json.load(fp=electoral_college_fp)

    electoral_college_df = pd.DataFrame.from_dict({'state': list(electoral_college.keys()),
                                                   'votes': list(electoral_college.values())})

    logger.info('Electoral College: {} total votes.'.format(electoral_college_df['votes'].sum()))
    url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'

    with open(file='./state_abbreviations.json', mode='r') as abbreviation_fp:
        state_abbreviations = json.load(fp=abbreviation_fp)

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

    review_2016_df = pd.read_csv('./world-population-review.csv')
    logger.info(list(review_2016_df))
    # patch up what DC is called here
    review_2016_df['State'] = review_2016_df['State'].replace(to_replace='Washington DC',
                                                              value='District of Columbia', )

    # patch up 2016 Main Congressional District votes
    review_2016_df = review_2016_df.append(
        {'State': 'Maine CD-1', 'votesDem': 212774, 'percD': 53.96, 'votesRep': 154384, 'percR': 39.15,
         'electoralDem': 1, 'electoralRep': 0, 'Pop': 394329, }, ignore_index=True)
    review_2016_df = review_2016_df.append(
        {'State': 'Maine CD-2', 'votesDem': 144817, 'percD': 40.98, 'votesRep': 181177, 'percR': 51.26,
         'electoralDem': 0, 'electoralRep': 1, 'Pop': 353416, }, ignore_index=True)
    review_2016_df = review_2016_df.append(
        {'State': 'Nebraska CD-1', 'votesDem': 100126, 'percD': 35.46, 'votesRep': 158626, 'percR': 56.18,
         'electoralDem': 0, 'electoralRep': 1, 'Pop': 282338, }, ignore_index=True)
    review_2016_df = review_2016_df.append(
        {'State': 'Nebraska CD-2', 'votesDem': 131030, 'percD': 44.92, 'votesRep': 137564, 'percR': 47.16,
         'electoralDem': 0, 'electoralRep': 1, 'Pop': 291680, }, ignore_index=True)
    review_2016_df = review_2016_df.append(
        {'State': 'Nebraska CD-3', 'votesDem': 53290, 'percD': 19.73, 'votesRep': 199657, 'percR': 73.92,
         'electoralDem': 0, 'electoralRep': 1, 'Pop': 270109, }, ignore_index=True)
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
    df = df[['question_id', 'state', 'end_date', 'answer', 'pct']]
    df = df[df.answer.isin({'Biden', 'Trump'})]
    df['question_id'] = df['question_id'].astype(int)

    a2_df = df[df.answer.isin({'Biden', 'Trump'})].groupby('question_id').filter(lambda x: len(x) == 2)
    cutoff_date = pd.Timestamp(datetime.datetime.today())
    biden_votes, trump_votes, ranked = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date, verbose=0)

    ranked = sorted(ranked, key=lambda x: abs(x[1]), reverse=True)
    ranked = [(rank[0], state_abbreviations[rank[0]], rank[1]) for rank in ranked]
    for rank in ranked:
        if rank[2] > 0:
            logger.info('state: {} margin: D+{:3.1f}'.format(rank[0], abs(rank[2])))
        elif rank[2] < 0:
            logger.info('state: {} margin: R+{:3.1f}'.format(rank[0], abs(rank[2])))
        else:
            logger.info('state: {} margin: 0.0'.format(rank[0]))

    logger.info([(rank[1], 'D' if rank[2] > 0 else 'R', abs(trunc(100.0 * rank[2]) / 100)) for rank in ranked if
                 abs(rank[2]) < 10.1])
    for limit in range(10, 0, -1):
        logger.info('there are {} states with margin {} percent or less'.format(
            sum([1 for rank in ranked if abs(rank[2]) <= limit]), limit))
    if 538 - biden_votes - trump_votes:
        logger.info('state: {} Biden: {} Trump: {} total: {} remaining: {}'.format('all', biden_votes, trump_votes,
                                                                                   biden_votes + trump_votes,
                                                                                   538 - biden_votes - trump_votes, ))
    else:
        logger.info('total: Biden: {} Trump: {}'.format(biden_votes, trump_votes, ))

    graph_df = pd.DataFrame(columns=['date', 'Biden', 'Trump', ])
    for cutoff_date in sorted(a2_df.end_date.unique(), ):
        biden_votes, trump_votes, _ = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                  verbose=0, )
        logger.info('date: {} Biden: {} Trump: {}'.format(cutoff_date, biden_votes, trump_votes, ))
        graph_df = graph_df.append({'date': cutoff_date, 'Biden': biden_votes, 'Trump': trump_votes, },
                                   ignore_index=True)

    fig, ax = plt.subplots(figsize=(15, 10))
    do_plot_matplotlib = False
    if do_plot_matplotlib:
        plt.scatter(x=graph_df.date, y=graph_df.Biden, c='b', )
        plt.scatter(x=graph_df.date, y=graph_df.Trump, c='r', )
        plt.savefig('./states-daily-matplotlib.png')
    else:
        graph_df['numbers'] = mdates.date2num(graph_df.date.values)
        sns.regplot(ax=ax, data=graph_df, x='numbers', y='Biden', )
        sns.regplot(ax=ax, data=graph_df, x='numbers', y='Trump', )
        plt.savefig('./states-daily-regplot.png')

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
