import datetime
from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import pandas as pd
import seaborn as sns
from matplotlib.dates import date2num
from matplotlib.dates import num2date
from matplotlib.pyplot import savefig
from matplotlib.pyplot import subplots
from numpy import poly1d
from numpy import polyfit

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO, )
    logger.info('started.', )

    url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'

    df = pd.read_csv(url, parse_dates=['end_date'], )

    logger.info(df.shape, )
    logger.info('\n{}'.format(df.head(5), ), )
    logger.info(list(df), )

    national_df = df[df.state.isnull()].copy(deep=True, )
    logger.info(national_df.shape)
    bt_df = national_df[national_df.answer.isin(['Biden', 'Trump'])].drop_duplicates()

    # drop data older than some arbitrary threshold
    bt_df = bt_df[bt_df['end_date'] > pd.Timestamp(datetime.date.today() - datetime.timedelta(weeks=20))]
    bt_df['poll_id'] = bt_df['poll_id'].astype(int)
    bt_df['question_id'] = bt_df['question_id'].astype(int)

    columns = {'pct': 'Biden', 'end_date': 'date', }
    biden_df = bt_df[bt_df.answer.isin(['Biden'])][['end_date', 'pct', 'question_id', ]].rename(columns=columns, )
    columns = {'pct': 'Trump', 'end_date': 'date', }
    trump_df = bt_df[bt_df.answer.isin(['Trump'])][['end_date', 'pct', 'question_id', ]].rename(columns=columns, )
    for key, value in {'Biden': biden_df, 'Trump': trump_df}.items():
        logger.info('we have {} rows of {} data'.format(len(value), key))

    # use the common question IDs to filter
    question_ids = sorted(list({item for item in biden_df.question_id.values if item in trump_df.question_id.values}))
    logger.info('we have {} common question IDs'.format(len(question_ids)))
    biden_df = biden_df[biden_df.question_id.isin(question_ids)][['date', 'Biden', ]]
    trump_df = trump_df[trump_df.question_id.isin(question_ids)][['date', 'Trump', ]]
    for key, value in {'Biden': biden_df, 'Trump': trump_df}.items():
        logger.info('we have {} rows of {} data'.format(len(value), key))

    fig0, ax0 = subplots()
    biden_df.set_index('date').plot(ax=ax0, c='blue', label='Biden', style='.', )
    # https://stackoverflow.com/questions/17638137/curve-fitting-to-a-time-series-in-the-format-datetime
    biden_date_numbers = date2num(biden_df.date.values, )
    for degree in range(1, 3):
        biden_fit = polyfit(x=biden_date_numbers, y=biden_df.Biden, deg=degree)
        biden_poly = poly1d(biden_fit)
        ax0.plot(num2date(biden_date_numbers, ), biden_poly(biden_date_numbers), 'b-')

    trump_df.set_index('date').plot(ax=ax0, c='red', label='Trump', style='.', )
    trump_date_numbers = date2num(trump_df.date.values, )
    for degree in range(1, 3):
        trump_fit = polyfit(x=trump_date_numbers, y=trump_df.Trump, deg=degree)
        trump_poly = poly1d(trump_fit)
        ax0.plot(num2date(trump_date_numbers), trump_poly(trump_date_numbers), 'r-', )

    scatter_png = './biden_trump_scatter.png'
    savefig(scatter_png)
    fig1, ax1 = subplots()

    # todo use mdates to turn this back into a date vs data graph
    biden_df['days'] = biden_df['date'].apply(lambda x: (x.to_pydatetime() - biden_df['date'].min()).days)
    trump_df['days'] = trump_df['date'].apply(lambda x: (x.to_pydatetime() - trump_df['date'].min()).days)
    sns.regplot(ax=ax1, data=biden_df, x='days', y='Biden', )
    sns.regplot(ax=ax1, data=trump_df, x='days', y='Trump', )
    regplot_png = './biden_trump_regplot.png'
    savefig(regplot_png)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
