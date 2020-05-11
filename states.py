import datetime
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import trunc
from time import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from get_data import get_data


def get_results(arg_df, arg_cutoff_date, electoral_df, historical_df, verbose, ):
    polling = {}
    arg_df = arg_df[arg_df.end_date <= arg_cutoff_date]
    for state in arg_df.state.unique():
        polling[state] = {}
        this_df = arg_df[arg_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        for candidate in [democrat, republican]:
            polling[state][candidate] = this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]
    result_democrat_votes, result_republican_votes = 0, 0
    result_ranked = list()
    for state in electoral_df.state.unique():
        if state in polling.keys():
            poll = polling[state]
            votes = electoral_df[electoral_df.state == state].votes.values[0]
            if poll[democrat] > poll[republican]:
                result_democrat_votes += votes
            elif poll[democrat] < poll[republican]:
                result_republican_votes += votes
            if poll[democrat] - poll[republican] > 0 and verbose:
                logger.info(
                    'state: {} polling margin: D+{:3.1f} pct'.format(state, abs(poll[democrat] - poll[republican])))
            elif poll[democrat] - poll[republican] < 0 and verbose:
                logger.info(
                    'state: {} polling margin: R+{:3.1f} pct'.format(state, abs(poll[democrat] - poll[republican])))
            else:
                if verbose:
                    logger.info('state: {} tied.'.format(state))
            result_ranked.append((state, poll[democrat] - poll[republican]))
        elif state in review_2016_df.State.unique():
            result_democrat_votes += historical_df[historical_df.State == state].electoralDem.values[0]
            result_republican_votes += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            if verbose:
                logger.warning('missing state: {}'.format(state))
    return result_democrat_votes, result_republican_votes, result_ranked


if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )
    register_matplotlib_converters()

    democrat = 'Biden'
    republican = 'Trump'
    electoral_college_df, review_2016_df, a2_df, state_abbreviations = get_data(democrat=democrat,
                                                                                republican=republican, )
    cutoff_date = pd.Timestamp(datetime.datetime.today())
    democrat_votes, republican_votes, ranked = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                           electoral_df=electoral_college_df,
                                                           historical_df=review_2016_df,
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
    if 538 - democrat_votes - republican_votes:
        format_string = 'state: {} {}: {} {}: {} total: {} remaining: {}'
        logger.info(format_string.format('all', democrat, democrat_votes, republican, republican_votes,
                                         democrat_votes + republican_votes, 538 - democrat_votes - republican_votes, ))
    else:
        logger.info('total: {}: {} {}: {}'.format(democrat, democrat_votes, republican, republican_votes, ))

    graph_df = pd.DataFrame(columns=['date', democrat, republican, ], )
    lm_df = pd.DataFrame(columns=['date', 'votes', 'candidate', ], )
    for cutoff_date in sorted(a2_df.end_date.unique(), ):
        democrat_votes, republican_votes, _ = get_results(arg_df=a2_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                          electoral_df=electoral_college_df,
                                                          historical_df=review_2016_df,
                                                          verbose=0, )
        logger.info(
            'date: {} {}: {} {}: {}'.format(pd.to_datetime(cutoff_date).date(), democrat, democrat_votes, republican,
                                            republican_votes, ))
        graph_df = graph_df.append(ignore_index=True,
                                   other={'date': cutoff_date, democrat: democrat_votes,
                                          republican: republican_votes, }, )
        lm_df = lm_df.append(ignore_index=True,
                             other={'date': cutoff_date, 'votes': democrat_votes, 'candidate': democrat, }, )
        lm_df = lm_df.append(ignore_index=True,
                             other={'date': cutoff_date, 'votes': republican_votes, 'candidate': republican, }, )

    lm_df['votes'] = lm_df['votes'].astype(float)
    lm_df['date'] = pd.to_datetime(lm_df['date'])
    lm_df['date'] = lm_df['date'].dt.date
    sns.set_style('darkgrid')
    plt.style.use('fivethirtyeight')
    plot_styles = ['lineplot', 'lmplot', 'matplotlib', 'pointplot', 'regplot', 'stategrid', 'swingstategrid']
    for plot_style in plot_styles:
        fig, ax = plt.subplots(figsize=(15, 10))
        if plot_style == plot_styles[0]:
            sns.lineplot(ax=ax, data=lm_df, hue='candidate', palette={democrat: 'b', republican: 'r'}, sort=True,
                         x='date', y='votes', )
            plt.savefig('./states-daily-lineplot.png', )
        elif plot_style == plot_styles[1]:
            lm_df['numbers'] = mdates.date2num(lm_df.date.values, )
            ax = sns.lmplot(data=lm_df, hue='candidate', order=3, palette={democrat: 'b', republican: 'r'}, x='numbers',
                            y='votes', ).set(xlim=(lm_df.numbers.min() - 100, lm_df.numbers.max() + 100,),
                                             ylim=(100, 450,), )
            plt.savefig('./states-daily-lmplot.png', )
        elif plot_style == plot_styles[2]:
            ax.scatter(x=graph_df.date, y=graph_df[democrat], c='b', )
            ax.scatter(x=graph_df.date, y=graph_df[republican], c='r', )
            plt.savefig('./states-daily-matplotlib.png', )
        elif plot_style == plot_styles[3]:
            # todo thin out the X axis so the ticks are readable
            ax = sns.pointplot(data=lm_df, hue='candidate', palette={democrat: 'b', republican: 'r'}, x='date',
                               y='votes', )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, )
            plt.savefig('./states-daily-pointplot.png', )
        elif plot_style == plot_styles[4]:
            graph_df['numbers'] = mdates.date2num(graph_df.date.values, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='numbers', y=democrat, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='numbers', y=democrat, lowess=True, scatter=False, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='numbers', y=democrat, logx=True, scatter=False, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='numbers', y=republican, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='numbers', y=republican, lowess=True, scatter=False, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='numbers', y=republican, logx=True, scatter=False, )
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
            # todo remove "state = " (?)
            plot = sns.FacetGrid(col='state', col_order=sorted(states), col_wrap=4, data=a3_df, hue='answer', )
            plot_result = plot.map(plt.scatter, 'date', 'pct', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=90, )
            plt.tight_layout()
            plt.savefig('./states-daily-swing-state-grid.png', )
        else:
            raise ValueError('plot style unknown.')

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
