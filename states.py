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
            trump_pct = poll['Trump']
            if abs(biden_pct - trump_pct) < 15.1:
                simulated_biden_result = binomial(n=1, p=biden_pct / (biden_pct + trump_pct))
            else:
                simulated_biden_result = int((1 + copysign(1, biden_pct - trump_pct)) / 2)
            result_biden_votes += votes * simulated_biden_result
            result_trump_votes += votes * (1 - simulated_biden_result)
        elif state in review_unique:
            result_biden_votes += historical_df[historical_df.State == state].electoralDem.values[0]
            result_trump_votes += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            logger.warning('missing state: {}'.format(state), )
    return result_biden_votes, result_trump_votes


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )
    register_matplotlib_converters()

    electoral_college_df, review_2016_df, a2_df, state_abbreviations = get_data()
    cutoff_date = pd.Timestamp(datetime.datetime.today())
    biden_votes, trump_votes, ranked = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                   electoral_df=electoral_college_df, historical_df=review_2016_df,
                                                   verbose=0, )

    ranked = sorted(ranked, key=lambda x: abs(x[1]), reverse=True, )
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
    realization_count = 1000
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
                                             ylim=(100, 450,), )
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
