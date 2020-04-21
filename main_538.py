from logging import INFO
from logging import basicConfig
from logging import getLogger
from time import time

import matplotlib.pyplot as plt
import pandas as pd

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
    biden_df = bt_df[bt_df.answer.isin(['Biden'])][['end_date', 'pct']].rename(columns={'pct': 'Biden'})
    trump_df = bt_df[bt_df.answer.isin(['Trump'])][['end_date', 'pct']].rename(columns={'pct': 'Trump'})
    fig, ax = plt.subplots()
    biden_df.set_index('end_date').plot(ax=ax, c='blue', label='Biden', style='.', )
    trump_df.set_index('end_date').plot(ax=ax, c='red', label='Trump', style='.', )
    plt.savefig('./biden_trump_scatter.png')

    logger.info('total time: {:5.2f}s'.format(time() - time_start))
