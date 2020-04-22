from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

if __name__ == '__main__':
    time_start = time()
    logger = getLogger(__name__)
    basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=INFO)
    logger.info('started.')

    url = 'https://projects.fivethirtyeight.com/polls-page/president_polls.csv'

    df = pd.read_csv(url, parse_dates=['end_date'])

    logger.info(df.shape)
    logger.info('\n{}'.format(df.head(5)))
    logger.info(list(df))

    national_df = df[df.state.isnull()].copy(deep=True)
    logger.info(national_df.shape)
    bt_df = national_df[national_df.answer.isin(['Biden', 'Trump'])].drop_duplicates()

    # drop data that is more than a year old
    bt_df = bt_df[bt_df['end_date'] > pd.Timestamp(datetime.date.today() - datetime.timedelta(days=366))]

    biden_df = bt_df[bt_df.answer.isin(['Biden'])][['end_date', 'pct']].rename(
        columns={'pct': 'Biden', 'end_date': 'date'}, )
    trump_df = bt_df[bt_df.answer.isin(['Trump'])][['end_date', 'pct']].rename(
        columns={'pct': 'Trump', 'end_date': 'date'}, )

    fig0, ax0 = plt.subplots()
    biden_df.set_index('date').plot(ax=ax0, c='blue', label='Biden', style='.', )
    # https://stackoverflow.com/questions/17638137/curve-fitting-to-a-time-series-in-the-format-datetime
    biden_date_numbers = mdates.date2num(biden_df.date.values)
    for degree in range(1, 4):
        biden_fit = np.polyfit(x=biden_date_numbers, y=biden_df.Biden, deg=degree)
        biden_poly = np.poly1d(biden_fit)
        ax0.plot(mdates.num2date(biden_date_numbers), biden_poly(biden_date_numbers), 'b-')

    trump_df.set_index('date').plot(ax=ax0, c='red', label='Trump', style='.', )
    trump_date_numbers = mdates.date2num(trump_df.date.values)
    for degree in range(1, 4):
        trump_fit = np.polyfit(x=trump_date_numbers, y=trump_df.Trump, deg=1)
        trump_poly = np.poly1d(trump_fit)
        ax0.plot(mdates.num2date(trump_date_numbers), trump_poly(trump_date_numbers), 'r-')

    scatter_png = './biden_trump_scatter.png'
    plt.savefig(scatter_png)
    fig1, ax1 = plt.subplots()

    biden_df['days'] = biden_df['date'].apply(lambda x: (x.to_pydatetime() - biden_df['date'].min()).days)
    trump_df['days'] = trump_df['date'].apply(lambda x: (x.to_pydatetime() - trump_df['date'].min()).days)
    sns.regplot(ax=ax1, data=biden_df, x='days', y='Biden', )
    sns.regplot(ax=ax1, data=trump_df, x='days', y='Trump', )
    regplot_png = './biden_trump_regplot.png'
    plt.savefig(regplot_png)

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
