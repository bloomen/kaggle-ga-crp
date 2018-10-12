import logging
import os
import pickle
import pandas as pd
from collections import OrderedDict
import json
from functools import partial
import numpy as np


logger = logging.getLogger(__name__)


NOT_SET = '(not set)'


def drop_column(df, name):
    logger.info('dropping: %s', name)
    df = df.drop([name], axis=1)
    logger.info('df.shape: %s', df.shape)
    return df


def make_class_target(y, y_max, n_classes):
    logger.info('making a classification target')
    y_max += 0.001
    return y.apply(lambda x: int(x / y_max * n_classes))


def flatten_data(df):
    logger.info('flatten data')

    json_cols = OrderedDict(
        device=OrderedDict(
            browser=None,
            operatingSystem=None,
            isMobile=None,
            deviceCategory=None, 
        ),
        geoNetwork=OrderedDict(
            continent=None,
            subContinent=None,
            country=None,
        ),
        totals=OrderedDict(
            visits=None,
            hits=None,
            pageviews='0',
            bounces='0',
            newVisits='0',
            transactionRevenue='0',
        ),
        trafficSource=OrderedDict(
            campaign=None,
            source=None,
            medium=None,
            keyword=NOT_SET,
            isTrueDirect=False,
            referralPath=NOT_SET,
            adContent=NOT_SET,
        )
    )

    new_cols = OrderedDict()

    def translate(x, col):
        data = json.loads(x)
        for key, default in json_cols[col].items():
            if default is None:
                val = data[key]
            else:
                val = data.get(key, default)
            key = col + '_' + key
            if key in new_cols:
                new_cols[key].append(val)
            else:
                new_cols[key] = [val]

    for col in json_cols.keys():
        df[col] = df[col].apply(partial(translate, col=col))
        df = drop_column(df, col)

    for key, val in new_cols.items():
        df[key] = pd.Series(val, index=df.index)

    logger.info('df.shape: %s', df.shape)
    return df


def drop_useless_columns(df):
    logger.info('dropping useless columns')

    useless_cols = [
        'socialEngagementType',  # there's only one type in the data
        'totals_visits',  # only one type
        'date',  # does the date of the visit matter?
        'sessionId',  # do we need to identify unique sessions?
        'visitId',  # do we need to identify unique visits?
        'visitStartTime',  # does it matter when the visit started?
    ]

    for col in useless_cols:
        df = drop_column(df, col)

    logger.info('df.shape: %s', df.shape)
    return df


def clean_data(df):
    logger.info('cleaning data')

    key = 'channelGrouping'
    logger.info(key)
    translator = {
        '(Other)': 0,
        'Organic Search': 1,
        'Social': 2,
        'Paid Search': 3,
        'Affiliates': 4,
        'Direct': 5,
        'Referral': 6,
        'Display': 7,
    }
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'trafficSource_campaign'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: 1 if x != NOT_SET else 0)

    key = 'trafficSource_keyword'
    logger.info(key)
    def keyword_index(x):
        if x == NOT_SET:
            return 0
        elif x == '(not provided)':
            return 0
        else:
            return 1
    df[key] = df[key].apply(
        lambda x: keyword_index(x))

    key = 'trafficSource_source'
    logger.info(key)
    def source_index(x):
        if x == NOT_SET:
            return 0
        elif x == '(direct)':
            return 1
        else:
            return 2
    df[key] = df[key].apply(
        lambda x: source_index(x))

    key = 'trafficSource_medium'
    logger.info(key)
    translator = {
        '(none)': 0,
        NOT_SET: 0,
        'organic': 1,
        'referral': 2,
        'cpc': 3,
        'affiliate': 4,
        'cpm': 5,
    }
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'trafficSource_isTrueDirect'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: 1 if x else 0)

    key = 'trafficSource_adContent'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: 1 if x != NOT_SET else 0)

    key = 'trafficSource_referralPath'
    logger.info(key)
    def referralPath_index(x):
        if x == NOT_SET:
            return 0
        elif x == '/':
            return 1
        else:
            return 2
    df[key] = df[key].apply(
        lambda x: referralPath_index(x))

    key = 'device_browser'
    logger.info(key)
    main_stream = ['Chrome', 'Firefox', 'Internet Explorer', 'Safari', 'Edge',
                   'IE', 'Android', 'Mozilla', 'Opera', 'Blackberry']
    def browser_index(x):
        for i, b in enumerate(main_stream):
            if b in x:
                return i + 1
        return 0
    df[key] = df[key].apply(
        lambda x: browser_index(x))

    key = 'device_isMobile'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: 1 if x else 0)

    key = 'device_operatingSystem'
    logger.info(key)
    translator = {
        NOT_SET: 0,
        'Android': 1,
        'iOS': 2,
        'Chrome OS': 3,
        'Windows Phone': 4,
        'Samsung': 5,
        'Xbox': 6,
        'Nintendo WiiU': 7,
        'Nintendo Wii': 7,
        'BlackBerry': 8,
        'Firefox OS': 9,
        'FreeBSD': 10,
        'OpenBSD': 10,
        'Nintendo 3DS': 11,
        'Nokia': 12,
        'NTT DoCoMo': 13,
        'SunOS': 14,
        'Macintosh': 15,
        'Windows':  16,
        'Linux': 17,
        'Tizen': 5,
        'SymbianOS': 5,
        'Playstation Vita': 6,
        'OS/2': 17,
    }
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'device_deviceCategory'
    logger.info(key)
    translator = {
        'desktop': 0,
        'tablet': 1,
        'mobile': 2,
    }
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'geoNetwork_country'
    logger.info(key)
    gdp = pd.read_csv('gdp_per_capita_2017.dat')
    translator = dict(zip(gdp['country'], gdp['gdp_per_capita']))
    translator[NOT_SET] = 17000
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'geoNetwork_continent'
    logger.info(key)
    translator = {
        NOT_SET: 0,
        'Asia': 1,
        'Oceania': 2,
        'Europe': 3,
        'Americas': 4,
        'Africa': 5,
    }
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'geoNetwork_subContinent'
    logger.info(key)
    translator = {
        NOT_SET: 0,
        'Western Asia': 1,
        'Australasia': 2,
        'Southern Europe': 3,
        'Southeast Asia': 4,
        'Northern Europe': 5,
        'Southern Asia': 6,
        'Western Europe': 7,
        'South America': 8,
        'Eastern Asia': 9,
        'Eastern Europe': 10,
        'Northern America': 11,
        'Western Africa': 12,
        'Central America': 13,
        'Eastern Africa': 14,
        'Caribbean': 15,
        'Southern Africa': 16,
        'Northern Africa': 17,
        'Central Asia': 18,
        'Middle Africa': 19,
        'Melanesia': 20,
        'Micronesian Region': 21,
        'Polynesia': 22,
    }
    df[key] = df[key].apply(
        lambda x: translator[x])

    key = 'totals_bounces'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: int(x))

    key = 'totals_hits'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: int(x))

    key = 'totals_newVisits'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: int(x))

    key = 'totals_pageviews'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: int(x))

    key = 'totals_transactionRevenue'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: np.log(float(x) + 1))

    logger.info('df.shape: %s', df.shape)
    return df


def load_data(data_type):
    logger.info('loading data')
    cache = data_type + '.pickle'
    if os.path.exists(cache):
        logger.info('reading cached data')
        with open(cache, 'rb') as f:
            data = pickle.load(f)
        logger.info('data.shape = %s', data.shape)
        return data
    else:
        logger.info('reading raw data')
        columns = OrderedDict(
            channelGrouping=str,
            date=str,
            device=str,
            fullVisitorId=str,
            geoNetwork=str,
            sessionId=str,
            socialEngangementType=str,
            totals=str,
            trafficSource=str,
            visitId=str,
            visitNumber=int,
            visitStartTime=int,
        )
        df = pd.read_csv(data_type + '.csv', dtype=columns)
        logger.info('df.shape = %s', df.shape)
        df = flatten_data(df)
        df = drop_useless_columns(df)
        df = clean_data(df)
        with open(cache, 'wb') as f:
            pickle.dump(df, f)
        return df
