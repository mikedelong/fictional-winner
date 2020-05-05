import datetime
import json
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import trunc
from time import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.random import binomial
from pandas.plotting import register_matplotlib_converters


def get_results(arg_df, arg_cutoff_date, electoral_df, historical_df, verbose, ):
    polling = {}
    arg_df = arg_df[arg_df.end_date <= arg_cutoff_date]
    for state in arg_df.state.unique():
        polling[state] = {}
        this_df = arg_df[arg_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        for candidate in ['Biden', 'Trump']:
            polling[state][candidate] = this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]
    result_biden_votes, result_trump_votes = 0, 0
    result_ranked = list()
    for state in electoral_df.state.unique():
        if state in polling.keys():
            poll = polling[state]
            votes = electoral_df[electoral_df.state == state].votes.values[0]
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
            result_biden_votes += historical_df[historical_df.State == state].electoralDem.values[0]
            result_trump_votes += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            if verbose:
                logger.warning('missing state: {}'.format(state))
    return result_biden_votes, result_trump_votes, result_ranked


def get_realization(arg_df, arg_cutoff_date, electoral_df, historical_df, ):
    polling = {}
    arg_df = arg_df[arg_df.end_date <= arg_cutoff_date]
    for state in arg_df.state.unique():
        polling[state] = {}
        this_df = arg_df[arg_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        for candidate in ['Biden', 'Trump']:
            polling[state][candidate] = this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]
    result_biden_votes = 0
    result_trump_votes = 0
    review_unique = review_2016_df.State.unique()
    for state in electoral_df.state.unique():
        if state in polling.keys():
            poll = polling[state]
            votes = electoral_df[electoral_df.state == state].votes.values[0]
            biden_pct = poll['Biden']
            simulated_biden_result = binomial(n=1, p=biden_pct / (biden_pct + poll['Trump']))
            result_biden_votes += votes * simulated_biden_result
            result_trump_votes += votes * (1 - simulated_biden_result)
        elif state in review_unique:
            result_biden_votes += historical_df[historical_df.State == state].electoralDem.values[0]
            result_trump_votes += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            logger.warning('missing state: {}'.format(state))
    return result_biden_votes, result_trump_votes


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

    with open(file='./state_abbreviations.json', mode='r') as abbreviation_fp:
        state_abbreviations = json.load(fp=abbreviation_fp)

    url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'
    df = pd.read_csv(url, parse_dates=['end_date'])
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
            logger.warning('no Electoral College data for {}'.format(item))
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
    df = df[df.answer.isin({'Biden', 'Trump'})]
    df['question_id'] = df['question_id'].astype(int)

    a2_df = df[df.answer.isin({'Biden', 'Trump'})].groupby('question_id').filter(lambda x: len(x) == 2)
    # filter out low-grade polls (?)
    a2_df = a2_df[a2_df.fte_grade.isin(['A+', 'A', 'A-', 'A/B', 'B', 'B-', 'B/C', 'C', ])]
    cutoff_date = pd.Timestamp(datetime.datetime.today())
    biden_votes, trump_votes, ranked = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                   electoral_df=electoral_college_df, historical_df=review_2016_df,
                                                   verbose=0, )

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

    realizations = list()
    realization_count = 100
    count_biden = 0
    count_trump = 0
    biden_realizations = list()
    for index, realization in enumerate(range(realization_count)):
        realization_biden, realization_trump = get_realization(arg_df=a2_df.copy(deep=True),
                                                               arg_cutoff_date=cutoff_date,
                                                               electoral_df=electoral_college_df,
                                                               historical_df=review_2016_df, )
        count_biden += 1 if realization_biden > realization_trump else 0
        count_trump += 1 if realization_biden < realization_trump else 0
        format_string = '{} Biden: {} Trump: {} Biden: {} Trump: {} ratio: {:5.4f} mean: {:5.1f} median: {}'
        biden_realizations = [item[0] for item in realizations]
        if len(biden_realizations):
            logger.info(format_string.format(index, realization_biden, realization_trump, count_biden, count_trump,
                                             count_biden / (count_biden + count_trump),
                                             np.array(biden_realizations).mean(),
                                             int(np.median(np.array(biden_realizations)), ), ))
        realizations.append((realization_biden, realization_trump,))
    bin_count = max(biden_realizations) - min(biden_realizations) + 1
    biden_win_realizations = [item for item in biden_realizations if item >= 270]
    biden_lose_realizations = [item for item in biden_realizations if item < 270]
    logger.info('Biden simulated wins: {} out of {} realizations'.format(
        len(biden_win_realizations), len(biden_realizations), len(biden_win_realizations) / len(biden_realizations), ))
    logger.info('Biden mean outcome: {:5.2f} median outcome: {}'.format(np.array(biden_realizations).mean(),
                                                                        np.median(np.array(biden_realizations))), )
    plt.hist(x=biden_win_realizations, bins=bin_count, color='blue', )
    plt.hist(x=biden_lose_realizations, bins=bin_count, color='red', )
    plt.savefig('./biden-histogram.png', )
    graph_df = pd.DataFrame(columns=['date', 'Biden', 'Trump', ], )
    lm_df = pd.DataFrame(columns=['date', 'votes', 'candidate', ], )
    for cutoff_date in sorted(a2_df.end_date.unique(), ):
        biden_votes, trump_votes, _ = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                  electoral_df=electoral_college_df, historical_df=review_2016_df,
                                                  verbose=0, )
        logger.info(
            'date: {} Biden: {} Trump: {}'.format(pd.to_datetime(cutoff_date).date(), biden_votes, trump_votes, ))
        graph_df = graph_df.append(ignore_index=True,
                                   other={'date': cutoff_date, 'Biden': biden_votes, 'Trump': trump_votes, }, )
        lm_df = lm_df.append(ignore_index=True,
                             other={'date': cutoff_date, 'votes': biden_votes, 'candidate': 'Biden', }, )
        lm_df = lm_df.append(ignore_index=True,
                             other={'date': cutoff_date, 'votes': trump_votes, 'candidate': 'Trump', }, )

    lm_df['votes'] = lm_df['votes'].astype(float)
    sns.set_style('darkgrid')
    plt.style.use('fivethirtyeight')
    plot_styles = ['lineplot', 'lmplot', 'matplotlib', 'pointplot', 'regplot', 'stategrid', 'swingstategrid']
    for plot_style in plot_styles:
        fig, ax = plt.subplots(figsize=(15, 10))
        if plot_style == plot_styles[0]:
            sns.lineplot(ax=ax, data=lm_df, hue='candidate', palette=dict(Biden='b', Trump='r'), sort=True, x='date',
                         y='votes', )
            plt.savefig('./states-daily-lineplot.png', )
        elif plot_style == plot_styles[1]:
            lm_df['numbers'] = mdates.date2num(lm_df.date.values, )
            ax = sns.lmplot(data=lm_df, hue='candidate', order=3, palette=dict(Biden='b', Trump='r', ), x='numbers',
                            y='votes', ).set(xlim=(lm_df.numbers.min() - 100, lm_df.numbers.max() + 100,),
                                             ylim=(100, 450), )
            plt.savefig('./states-daily-lmplot.png', )
        elif plot_style == plot_styles[2]:
            ax.scatter(x=graph_df.date, y=graph_df.Biden, c='b', )
            ax.scatter(x=graph_df.date, y=graph_df.Trump, c='r', )
            plt.savefig('./states-daily-matplotlib.png', )
        elif plot_style == plot_styles[3]:
            ax = sns.pointplot(data=lm_df, hue='candidate', palette=dict(Biden='b', Trump='r', ), x='date', y='votes', )
            plt.savefig('./states-daily-pointplot.png', )
        elif plot_style == plot_styles[4]:
            graph_df['numbers'] = mdates.date2num(graph_df.date.values, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='numbers', y='Biden', )
            sns.regplot(ax=ax, color='b', data=graph_df, x='numbers', y='Biden', lowess=True, scatter=False, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='numbers', y='Biden', logx=True, scatter=False, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='numbers', y='Trump', )
            sns.regplot(ax=ax, color='r', data=graph_df, x='numbers', y='Trump', lowess=True, scatter=False, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='numbers', y='Trump', logx=True, scatter=False, )
            plt.savefig('./states-daily-regplot.png', )
        elif plot_style == plot_styles[5]:
            plot = sns.FacetGrid(col='state', col_order=sorted(a2_df.state.unique()), col_wrap=6, data=a2_df,
                                 hue='answer', )
            plot_result = plot.map(plt.scatter, 'end_date', 'pct', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90, )
            plt.tight_layout()
            plt.savefig('./states-daily-state-grid.png', )
        elif plot_style == plot_styles[6]:
            states = [state for state in a2_df.state.unique() if a2_df.state.value_counts()[state] > 7]
            a3_df = a2_df[a2_df.state.isin(states)].copy(deep=True)
            a3_df['date'] = [datetime.datetime.date(item) for item in a3_df['end_date']]
            plot = sns.FacetGrid(col='state', col_order=sorted(states), col_wrap=4, data=a3_df, hue='answer', )
            plot_result = plot.map(plt.scatter, 'date', 'pct', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90, )
            plt.tight_layout()
            plt.savefig('./states-daily-swing-state-grid.png', )
        else:
            raise ValueError('plot style unknown.')

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
