import datetime
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import sqrt
from math import trunc
from time import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from seaborn import lineplot
from seaborn import lmplot
from seaborn import pointplot
from seaborn import set_style
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
    grade_filter = {'A+', 'A', 'A-', 'A/B', 'B', 'B-', 'B/C', 'C', }

    electoral_college_df, review_2016_df, data_df, state_abbreviations = get_data(democrat=democrat,
                                                                                  grade_to_filter=grade_filter,
                                                                                  republican=republican, )
    cutoff_date = pd.Timestamp(datetime.datetime.today())
    democrat_votes, republican_votes, ranked = get_results(
        arg_df=data_df.copy(deep=True), arg_cutoff_date=cutoff_date, electoral_df=electoral_college_df,
        historical_df=review_2016_df, verbose=0, )

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
    for cutoff_date in sorted(data_df.end_date.unique(), ):
        democrat_votes, republican_votes, _ = get_results(arg_df=data_df.copy(deep=True), arg_cutoff_date=cutoff_date,
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
    set_style('darkgrid')
    plt.style.use('fivethirtyeight')
    plot_styles = ['lineplot', 'lmplot', 'matplotlib', 'pointplot', 'regplot', 'stategrid', 'swingstategrid',
                   'staterank', ]
    figsize = (15, 10)
    palette = {democrat: 'b', republican: 'r'}
    rotation = 60
    for plot_style in plot_styles:
        fig, ax = plt.subplots(figsize=figsize)
        if plot_style == plot_styles[0]:
            lineplot(ax=ax, data=lm_df, hue='candidate', palette=palette, sort=True,
                         x='date', y='votes', )
            lineplot_png = './states-lineplot.png'
            logger.info('saving {} to {}'.format(plot_style, lineplot_png, ), )
            plt.savefig(lineplot_png, )
        elif plot_style == plot_styles[1]:
            lm_df['date'] = mdates.date2num(lm_df.date.values, )
            ax = lmplot(data=lm_df, hue='candidate', order=3, palette=palette, x='date', y='votes', ).set(
                xlim=(lm_df.date.min() - 100, lm_df.date.max() + 100,), ylim=(100, 450,), )
            ax.set_xticklabels(labels=[mdates.num2date(number, tz=None, ).date() for number in lm_df.date.values], )
            lmplot_png = './states-daily-lmplot.png'
            logger.info('saving {} to {}'.format(plot_style, lmplot_png, ), )
            plt.savefig(lmplot_png, )
        elif plot_style == plot_styles[2]:
            ax.scatter(c='b', x=graph_df.date, y=graph_df[democrat], )
            ax.scatter(c='r', x=graph_df.date, y=graph_df[republican], )
            matplotlib_png = './states-historical-scatter.png'
            logger.info('saving {} to {}'.format(plot_style, matplotlib_png, ), )
            plt.savefig(matplotlib_png, )
        elif plot_style == plot_styles[3]:
            # todo thin out the X axis so the ticks are readable
            pointplot(ax=ax, data=lm_df, hue='candidate', palette=palette, x='date', y='votes', )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, )
            pointplot_png = './states-historical-pointplot.png'
            logger.info('saving {} to {}'.format(plot_style, pointplot_png, ), )
            plt.savefig(pointplot_png, )
        elif plot_style == plot_styles[4]:
            graph_df['date'] = mdates.date2num(graph_df.date.values, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='date', y=democrat, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='date', y=democrat, lowess=True, scatter=False, )
            sns.regplot(ax=ax, color='b', data=graph_df, x='date', y=democrat, logx=True, scatter=False, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='date', y=republican, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='date', y=republican, lowess=True, scatter=False, )
            sns.regplot(ax=ax, color='r', data=graph_df, x='date', y=republican, logx=True, scatter=False, )
            ax.set_xticklabels(labels=[mdates.num2date(number, tz=None, ).date() for number in lm_df.date.values], )
            regplot_png = './states-daily-regplot.png'
            logger.info('saving {} to {}'.format(plot_style, regplot_png, ), )
            plt.savefig(regplot_png, )
        elif plot_style == plot_styles[5]:
            col_wrap = int(sqrt(data_df.state.nunique()))
            data_df = data_df.rename(columns={'end_date': 'date', 'pct': 'percent', }, )
            plot = sns.FacetGrid(col='state', col_order=sorted(data_df.state.unique()), col_wrap=col_wrap, data=data_df,
                                 hue='answer', )
            plot_result = plot.map(plt.scatter, 'date', 'percent', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ))
            plt.tight_layout()
            state_grid_png = './states-daily-state-grid.png'
            logger.info('saving {} to {}'.format(plot_style, state_grid_png, ), )
            plt.savefig(state_grid_png, )
            differences = dict()
            for question in data_df['question_id'].unique():
                difference_df = data_df[data_df['question_id'] == question]
                differences[question] = difference_df[difference_df['answer'] == democrat]['percent'].values[0] - \
                                        difference_df[difference_df['answer'] == republican]['percent'].values[0]
            # now that we have the question-difference dict let's build a DataFrame we can use to make the FacetGrid
            grid_df = data_df[['question_id', 'date', 'state', ]].drop_duplicates()
            grid_df['difference'] = grid_df['question_id'].map(differences)
            states = [state for state in data_df.state.unique() if data_df.state.value_counts()[state] > 2]
            grid_df = grid_df[grid_df['state'].isin(states)]
            plot = sns.FacetGrid(col='state', col_order=sorted(grid_df.state.unique()), col_wrap=col_wrap,
                                 data=grid_df, )
            plot_result = plot.map(plt.plot, 'date', 'difference', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ))
            plt.tight_layout()
            state_plot_png = './states-daily-state-plot.png'
            logger.info('saving {} to {}'.format(plot_style, state_plot_png, ), )
            plt.savefig(state_plot_png, )
        elif plot_style == plot_styles[6]:
            states = [state for state in data_df.state.unique() if data_df.state.value_counts()[state] > 8]
            swing_df = data_df[data_df.state.isin(states)].copy(deep=True)
            swing_df['date'] = [datetime.datetime.date(item) for item in swing_df['date']]
            swing_df = swing_df.rename(columns={'pct': 'percent', }, )
            col_wrap = int(sqrt(swing_df.state.nunique()))
            plot = sns.FacetGrid(col='state', col_order=sorted(states), col_wrap=col_wrap, data=swing_df,
                                 hue='answer', )
            plot_result = plot.map(plt.scatter, 'date', 'percent', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ))
            plt.tight_layout()
            swing_state_grid_png = './states-daily-swing-state-grid.png'
            logger.info('saving {} to {}'.format(plot_style, swing_state_grid_png, ), )
            plt.savefig(swing_state_grid_png, )
            differences = dict()
            for question in swing_df['question_id'].unique():
                difference_df = swing_df[swing_df['question_id'].isin({question})]
                differences[question] = difference_df[difference_df['answer'] == democrat]['percent'].values[0] - \
                             difference_df[difference_df['answer'] == republican]['percent'].values[0]
            # now that we have the question-difference dict let's build a DataFrame we can use to make the FacetGrid
            grid_df = swing_df[['date', 'question_id', 'state', ]].drop_duplicates()
            grid_df['difference'] = grid_df['question_id'].map(differences)
            plot = sns.FacetGrid(col='state', col_order=sorted(grid_df.state.unique()), col_wrap=col_wrap,
                                 data=grid_df, )
            plot_result = plot.map(plt.plot, 'date', 'difference', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ))
            plt.tight_layout()
            state_plot_png = './states-daily-swing-plot.png'
            logger.info('saving {} to {}'.format(plot_style, state_plot_png, ), )
            plt.savefig(state_plot_png, )
        elif plot_style == plot_styles[7]:
            rank_df = pd.DataFrame([(rank[1], rank[2]) for rank in ranked], columns=['State', 'margin', ], )
            rank_df['abs_margin'] = rank_df['margin'].abs()
            rank_df['color'] = rank_df['margin'].apply(lambda x: 'r' if x <= 0 else 'b')
            rank_df['candidate'] = rank_df['margin'].apply(lambda x: republican if x <= 0 else democrat)
            figure = plt.figure(figsize=figsize)
            for index, rank in enumerate(ranked):
                logger.info(rank)
                plt.scatter(x=rank_df.index, y=rank_df.abs_margin, c=rank_df.color, )
            rank_png = './state-rank.png'
            logger.info('saving {} to {}'.format(plot_style, rank_png, ), )
            plt.savefig(rank_png, )
            del figure
            figure = plt.figure(figsize=figsize)
            ax_scatter = sns.scatterplot(data=rank_df, hue='candidate', x='State', y='abs_margin', )
            rank_scatterplot_png = './state-rank-scatterplot.png'
            logger.info('saving {} to {}'.format(plot_style, rank_scatterplot_png, ), )
            plt.savefig(rank_scatterplot_png, )
            del figure
            figure = plt.figure(figsize=figsize)
            ax_bar = sns.barplot(data=rank_df, hue='candidate', x='State', y='abs_margin', )
            rank_barplot_png = './state-rank-barplot.png'
            logger.info('saving {} to {}'.format(plot_style, rank_barplot_png, ), )
            plt.savefig(rank_barplot_png, )
        else:
            raise ValueError('plot style unknown.')

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
