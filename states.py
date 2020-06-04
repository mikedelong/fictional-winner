from datetime import datetime
from logging import INFO
from logging import basicConfig
from logging import getLogger
from math import sqrt
from math import trunc
from time import time

from matplotlib.dates import date2num
from matplotlib.dates import num2date
from matplotlib.pyplot import figure
from matplotlib.pyplot import locator_params
from matplotlib.pyplot import plot as matplotlib_plot
from matplotlib.pyplot import savefig
from matplotlib.pyplot import scatter
from matplotlib.pyplot import style
from matplotlib.pyplot import subplots
from matplotlib.pyplot import tight_layout
from pandas import DataFrame
from pandas import Timestamp
from pandas import to_datetime
from pandas.plotting import register_matplotlib_converters
from seaborn import FacetGrid
from seaborn import barplot
from seaborn import lineplot
from seaborn import lmplot
from seaborn import pointplot
from seaborn import regplot
from seaborn import scatterplot
from seaborn import set_style

from get_data import get_data


def get_results(arg_df, arg_cutoff_date, electoral_df, historical_df, verbose, ):
    polling = {}
    arg_df = arg_df[arg_df.end_date <= arg_cutoff_date]
    for state in arg_df.state.unique():
        this_df = arg_df[arg_df.state == state]
        this_df = this_df[this_df.end_date == this_df.end_date.max()]
        polling[state] = {candidate: this_df[this_df.answer.isin({candidate})].groupby('pct').mean().index[0]
                          for candidate in [democrat, republican]}
    result_democrat, result_republican = 0, 0
    result_ranked = list()
    unique_states = electoral_df.state.unique()
    polling_keys = polling.keys()
    for state in unique_states:
        if state in polling_keys:
            poll = polling[state]
            votes = electoral_df[electoral_df.state == state].votes.values[0]
            if poll[democrat] > poll[republican]:
                result_democrat += votes
            elif poll[democrat] < poll[republican]:
                result_republican += votes
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
            result_democrat += historical_df[historical_df.State == state].electoralDem.values[0]
            result_republican += historical_df[historical_df.State == state].electoralRep.values[0]
        else:
            if verbose:
                logger.warning('missing state: {}'.format(state))
    return result_democrat, result_republican, result_ranked


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
    cutoff_date = Timestamp(datetime.today())
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

    graph_df = DataFrame(columns=['date', democrat, republican, ], )
    lm_df = DataFrame(columns=['date', 'votes', 'candidate', ], )
    for cutoff_date in sorted(data_df.end_date.unique(), ):
        democrat_votes, republican_votes, _ = get_results(arg_df=data_df.copy(deep=True), arg_cutoff_date=cutoff_date,
                                                          electoral_df=electoral_college_df,
                                                          historical_df=review_2016_df, verbose=0, )
        logger.info(
            'date: {} {}: {} {}: {}'.format(to_datetime(cutoff_date).date(), democrat, democrat_votes, republican,
                                            republican_votes, ))
        graph_df = graph_df.append(ignore_index=True,
                                   other={'date': cutoff_date, democrat: democrat_votes,
                                          republican: republican_votes, }, )
        lm_df = lm_df.append(ignore_index=True,
                             other={'date': cutoff_date, 'votes': democrat_votes, 'candidate': democrat, }, )
        lm_df = lm_df.append(ignore_index=True,
                             other={'date': cutoff_date, 'votes': republican_votes, 'candidate': republican, }, )

    lm_df['votes'] = lm_df['votes'].astype(float)
    lm_df['date'] = to_datetime(lm_df['date']).dt.date
    set_style('darkgrid')
    style.use('fivethirtyeight')
    plot_styles = ['lineplot', 'lmplot', 'matplotlib', 'pointplot', 'regplot', 'stategrid', 'swingstategrid',
                   'staterank', ]
    figsize = (15, 10,)
    palette = {democrat: 'b', republican: 'r', }
    rotation = 60
    for plot_style in plot_styles:
        fig, ax = subplots(figsize=figsize, )
        if plot_style == plot_styles[0]:
            lineplot(ax=ax, data=lm_df, hue='candidate', palette=palette, sort=True, x='date', y='votes', )
            lineplot_png = './states-lineplot.png'
            logger.info('saving {} to {}'.format(plot_style, lineplot_png, ), )
            savefig(lineplot_png, )
        elif plot_style == plot_styles[1]:
            lm_df['date'] = date2num(lm_df.date.values, )
            ax = lmplot(data=lm_df, hue='candidate', order=3, palette=palette, x='date', y='votes', ).set(
                xlim=(lm_df.date.min() - 100, lm_df.date.max() + 100,), ylim=(100, 450,), )
            ax.set_xticklabels(labels=[num2date(number, tz=None, ).date() for number in lm_df.date.values], )
            lmplot_png = './states-daily-lmplot.png'
            logger.info('saving {} to {}'.format(plot_style, lmplot_png, ), )
            savefig(lmplot_png, )
        elif plot_style == plot_styles[2]:
            ax.scatter(c='b', x=graph_df.date, y=graph_df[democrat], )
            ax.scatter(c='r', x=graph_df.date, y=graph_df[republican], )
            matplotlib_png = './states-historical-scatter.png'
            logger.info('saving {} to {}'.format(plot_style, matplotlib_png, ), )
            savefig(matplotlib_png, )
        elif plot_style == plot_styles[3]:
            pointplot(ax=ax, data=lm_df, hue='candidate', palette=palette, x='date', y='votes', )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, )
            locator_params(axis='x', nbins=10, )
            pointplot_png = './states-historical-pointplot.png'
            logger.info('saving {} to {}'.format(plot_style, pointplot_png, ), )
            savefig(pointplot_png, )
        elif plot_style == plot_styles[4]:
            graph_df['date'] = date2num(graph_df.date.values, )
            regplot(ax=ax, color='b', data=graph_df, x='date', y=democrat, )
            regplot(ax=ax, color='b', data=graph_df, lowess=True, x='date', y=democrat, scatter=False, )
            regplot(ax=ax, color='b', data=graph_df, logx=True, x='date', y=democrat, scatter=False, )
            regplot(ax=ax, color='r', data=graph_df, x='date', y=republican, )
            regplot(ax=ax, color='r', data=graph_df, lowess=True, x='date', y=republican, scatter=False, )
            regplot(ax=ax, color='r', data=graph_df, logx=True, x='date', y=republican, scatter=False, )
            ax.set_xticklabels(labels=[num2date(number, tz=None, ).date() for number in lm_df.date.values], )
            regplot_png = './states-daily-regplot.png'
            logger.info('saving {} to {}'.format(plot_style, regplot_png, ), )
            savefig(regplot_png, )
        elif plot_style == plot_styles[5]:
            col_wrap = int(sqrt(data_df.state.nunique(), ), )
            data_df = data_df.rename(columns={'end_date': 'date', 'pct': 'percent', }, )
            plot = FacetGrid(col='state', col_order=sorted(data_df.state.unique()), col_wrap=col_wrap, data=data_df,
                             hue='answer', )
            plot_result = plot.map(scatter, 'date', 'percent', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ), )
            tight_layout()
            state_grid_png = './states-daily-state-grid.png'
            logger.info('saving {} to {}'.format(plot_style, state_grid_png, ), )
            savefig(state_grid_png, )
            differences = dict()
            for question in data_df['question_id'].unique():
                difference_df = data_df[data_df['question_id'] == question]
                left = difference_df[difference_df['answer'] == democrat]['percent'].values[0]
                right = difference_df[difference_df['answer'] == republican]['percent'].values[0]
                differences[question] = left - right
            # now that we have the question-difference dict let's build a DataFrame we can use to make the FacetGrid
            grid_df = data_df[['question_id', 'date', 'state', ]].drop_duplicates()
            grid_df['difference'] = grid_df['question_id'].map(differences)
            states = [state for state in data_df.state.unique() if data_df.state.value_counts()[state] > 2]
            grid_df = grid_df[grid_df['state'].isin(states)]
            plot = FacetGrid(col='state', col_order=sorted(grid_df.state.unique()), col_wrap=col_wrap,
                             data=grid_df, )
            plot_result = plot.map(matplotlib_plot, 'date', 'difference', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ))
            tight_layout()
            state_plot_png = './states-daily-state-plot.png'
            logger.info('saving {} to {}'.format(plot_style, state_plot_png, ), )
            savefig(state_plot_png, )
        elif plot_style == plot_styles[6]:
            states = [state for state in data_df.state.unique() if data_df.state.value_counts()[state] > 8]
            swing_df = data_df[data_df.state.isin(states)].copy(deep=True)
            swing_df['date'] = [datetime.date(item) for item in swing_df['date']]
            swing_df = swing_df.rename(columns={'pct': 'percent', }, )
            col_wrap = int(sqrt(swing_df.state.nunique()))
            plot = FacetGrid(col='state', col_order=sorted(states), col_wrap=col_wrap, data=swing_df, hue='answer', )
            plot_result = plot.map(scatter, 'date', 'percent', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ))
            tight_layout()
            swing_state_grid_png = './states-daily-swing-state-grid.png'
            logger.info('saving {} to {}'.format(plot_style, swing_state_grid_png, ), )
            savefig(swing_state_grid_png, )
            differences = dict()
            for question in swing_df['question_id'].unique():
                difference_df = swing_df[swing_df['question_id'].isin({question})]
                left = difference_df[difference_df['answer'] == democrat]['percent'].values[0]
                right = difference_df[difference_df['answer'] == republican]['percent'].values[0]
                differences[question] = left - right
            # now that we have the question-difference dict let's build a DataFrame we can use to make the FacetGrid
            grid_df = swing_df[['date', 'question_id', 'state', ]].drop_duplicates()
            grid_df['difference'] = grid_df['question_id'].map(differences)
            plot = FacetGrid(col='state', col_order=sorted(grid_df.state.unique()), col_wrap=col_wrap, data=grid_df, )
            plot_result = plot.map(matplotlib_plot, 'date', 'difference', )
            for axes in plot.axes.flat:
                _ = axes.set_xticklabels(axes.get_xticklabels(), rotation=rotation, )
            for axes in plot.axes.flatten():
                axes.set_title(axes.get_title().replace('state = ', '', ), )
            tight_layout()
            state_plot_png = './states-daily-swing-plot.png'
            logger.info('saving {} to {}'.format(plot_style, state_plot_png, ), )
            savefig(state_plot_png, )
        elif plot_style == plot_styles[7]:
            rank_df = DataFrame([(rank[1], rank[2]) for rank in ranked], columns=['State', 'margin', ], )
            rank_df['abs_margin'] = rank_df['margin'].abs()
            rank_df['color'] = rank_df['margin'].apply(lambda x: 'r' if x <= 0 else 'b')
            rank_df['candidate'] = rank_df['margin'].apply(lambda x: republican if x <= 0 else democrat)
            rank_figure = figure(figsize=figsize)
            for index, rank in enumerate(ranked):
                logger.info(rank)
                scatter(x=rank_df.index, y=rank_df.abs_margin, c=rank_df.color, )
            rank_png = './state-rank.png'
            logger.info('saving {} to {}'.format(plot_style, rank_png, ), )
            savefig(rank_png, )
            del rank_figure
            scatter_figure = figure(figsize=figsize)
            ax_scatter = scatterplot(data=rank_df, hue='candidate', x='State', y='abs_margin', )
            rank_scatterplot_png = './state-rank-scatterplot.png'
            logger.info('saving {} to {}'.format(plot_style, rank_scatterplot_png, ), )
            savefig(rank_scatterplot_png, )
            del scatter_figure
            bar_figure = figure(figsize=figsize)
            ax_bar = barplot(data=rank_df, hue='candidate', x='State', y='abs_margin', )
            rank_barplot_png = './state-rank-barplot.png'
            logger.info('saving {} to {}'.format(plot_style, rank_barplot_png, ), )
            savefig(rank_barplot_png, )
        else:
            raise ValueError('plot style unknown.')

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
