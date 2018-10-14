import logging
import os
import pickle
import pandas as pd
from collections import OrderedDict
import json
from functools import partial
import numpy as np
from datetime import datetime
import requests
import time
from dateutil import tz
import pytz


logger = logging.getLogger(__name__)


NOT_SET = '(not set)'


def get_timezone(lat, lng):
    logger.info('get timezone for: %f, %f', lat, lng)
    params = {
        'key': 'API_KEY',
        'by': 'position',
        'format': 'json',
        'lat': str(lat),
        'lng': str(lng),
    }
    res = requests.get("http://api.timezonedb.com/v2.1/get-time-zone", params)
    if res.status_code != 200:
        raise RuntimeError(res.status_code)
    time.sleep(1.1)
    return json.loads(res.text)['zoneName']


def drop_column(df, name):
    logger.info('dropping: %s', name)
    df = df.drop([name], axis=1)
    logger.info('df.shape: %s', df.shape)
    return df


def make_class_target(y, n_classes):
    logger.info('making a classification target')
    d = []
    for val in y:
        if val > 0:
            d.append(val)
    quants = np.quantile(d, np.linspace(0, 1, n_classes))
    quants[-1] += 0.001
    for i in range(y.shape[0]):
        val = y.iloc[i]
        if val == 0:
            continue
        for j in range(quants.shape[0]-1):
            if val >= quants[j] and val < quants[j+1]:
                y.iloc[i] = j + 1
                break
    logger.info('y.shape: %s', y.shape)
    logger.info('quants.shape: %s', quants.shape)
    return y, quants


def make_class_target2(y, y_max, n_classes):
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
            city=None,
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
        'visitId',  # seems to be similar to sessionId
        'geoNetwork_continent',  # use gdp instead
        'geoNetwork_subContinent',  # use gdp instead
        'device_operatingSystem',  # don't expect to matter
        'date',  # use visitStartTime instead
    ]

    for col in useless_cols:
        df = drop_column(df, col)

    logger.info('df.shape: %s', df.shape)
    return df


def one_hot_encode(df, column, names):
    logger.info('one hot encoding: %s', column)
    size = df[column].shape[0]
    data = OrderedDict()
    for n in names:
        data[n] = [0]*size
    for i, value in enumerate(df[column]):
        if value in data:
            data[value][i] = 1
    for key, val in data.items():
        df[column + '_' + str(key)] = val
    df = drop_column(df, column)
    logger.info('df.shape: %s', df.shape)
    return df


def generate_timezones(df):
    logger.info('generating timezones')
    world = pd.read_csv('worldcities.csv')

    timezone_country_cache = "timezone_country.pickle"
    if not os.path.exists(timezone_country_cache):
        timezone_country = {}  # country -> timezone
        for country in df['geoNetwork_country']:
            if country == NOT_SET or country in timezone_country:
                continue
            logger.info('country: %s', country)
            series = world.loc[world['country'] == country]
            if series.shape[0] > 0:
                sub = series.loc[series['capital'] == 'primary']
                if sub.shape[0] > 0:
                    series = sub.iloc[0]
                else:
                    series = series.iloc[0]
                timezone_country[country] = get_timezone(series['lat'], series['lng'])
            else:
                logger.info("missing country: %s", country)
        with open(timezone_country_cache, 'wb') as f:
            pickle.dump(timezone_country, f)
    else:
        logger.info('using cache: %s', timezone_country_cache)
        with open(timezone_country_cache, 'rb') as f:
            timezone_country = pickle.load(f)

    timezone_city_cache = "timezone_city.pickle"
    if not os.path.exists(timezone_city_cache):
        timezone_city = {}  # city -> timezone
        for city, country in zip(df['geoNetwork_city'], df['geoNetwork_country']):
            key = city + '_' + country
            logger.info('key: %s', key)
            if city == 'not available in demo dataset' or key in timezone_city:
                continue
            series = world.loc[(world['city'] == city) & (world['country'] == country)]
            if series.shape[0] > 0:
                series = series.iloc[0]
                timezone_city[key] = get_timezone(series['lat'], series['lng'])
        with open(timezone_city_cache, 'wb') as f:
            pickle.dump(timezone_city, f)
    else:
        logger.info('using cache: %s', timezone_city_cache)
        with open(timezone_city_cache, 'rb') as f:
            timezone_city = pickle.load(f)

    return timezone_city, timezone_country


def clean_data(df, timezone_city, timezone_country):
    logger.info('cleaning data')

    key = 'visitStartTime'
    logger.info(key)
    weekday = []
    month = []
    morning = [0]*df.shape[0]
    lunchtime = [0]*df.shape[0]
    daytime = [0]*df.shape[0]
    evening = [0]*df.shape[0]
    nighttime = [0]*df.shape[0]
    i = 0
    for city, country, atime in zip(df['geoNetwork_city'],
                                    df['geoNetwork_country'],
                                    df['visitStartTime']):
        atime = datetime.utcfromtimestamp(int(atime))
        label = city + '_' + country
        if label in timezone_city:
            timezone = timezone_city[label]
        else:
            timezone = timezone_country.get(country)

        if timezone is not None:
            atime = pytz.timezone(timezone).localize(atime)
            hour = atime.hour
        else:
            hour = np.random.random() * 24

        if hour >= 0 and hour < 6:
            nighttime[i] = 1
        elif hour >= 6 and hour < 8:
            morning[i] = 1
        elif hour >= 12 and hour < 14:
            lunchtime[i] = 1
        elif (hour >= 8 and hour < 12) or (hour >= 14 and hour < 18):
            daytime[i] = 1
        else:
            evening[i] = 1
        weekday.append(atime.weekday())
        month.append(atime.month)
        i += 1

    df = drop_column(df, 'geoNetwork_city')
    df = drop_column(df, key)
    df[key + '_morning'] = morning
    df[key + '_lunchtime'] = lunchtime
    df[key + '_daytime'] = daytime
    df[key + '_evening'] = evening
    df[key + '_nighttime'] = nighttime
    df[key + '_month'] = month
    df[key + '_weekday'] = weekday
    df = one_hot_encode(df, key + '_month', range(1, 13))
    df = one_hot_encode(df, key + '_weekday', range(0, 7))

    key = 'channelGrouping'
    logger.info(key)
    names = [
        '(Other)',
        'Organic Search',
        'Social',
        'Paid Search',
        'Affiliates',
        'Direct',
        'Referral',
        'Display',
    ]
    df = one_hot_encode(df, key, names)

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
            return 2
        else:
            return 1
    df[key] = df[key].apply(
        lambda x: source_index(x))

    key = 'trafficSource_medium'
    logger.info(key)
    names = [
        NOT_SET,
        'organic',
        'referral',
        'cpc',
        'affiliate',
        'cpm',
    ]
    df[key] = df[key].apply(lambda x: NOT_SET if x == '(none)' else x)
    df = one_hot_encode(df, key, names)

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
        else:
            return 1
    df[key] = df[key].apply(
        lambda x: referralPath_index(x))

    key = 'device_browser'
    logger.info(key)
    main_stream = ['Chrome', 'Firefox', 'Internet Explorer', 'Safari', 'Edge',
                   'IE', 'Android', 'Mozilla', 'Blackberry']
    df[key] = df[key].apply(lambda x: 1 if x in main_stream else 0)

    key = 'device_isMobile'
    logger.info(key)
    df[key] = df[key].apply(
        lambda x: 1 if x else 0)

#     key = 'device_operatingSystem'
#     logger.info(key)
#     names = [
#         NOT_SET,
#         'Android',
#         'iOS',
#         'Chrome OS',
#         'Windows Phone',
#         'Samsung',
#         'Xbox',
#         'Nintendo WiiU',
#         'Nintendo Wii',
#         'BlackBerry',
#         'Firefox OS',
#         'FreeBSD',
#         'OpenBSD',
#         'Nintendo 3DS',
#         'Nokia',
#         'NTT DoCoMo',
#         'SunOS',
#         'Macintosh',
#         'Windows',
#         'Linux',
#         'Tizen',
#         'SymbianOS',
#         'Playstation Vita',
#         'OS/2',
#     ]
#     df = one_hot_encode(df, key, names)

    key = 'device_deviceCategory'
    logger.info(key)
    names = [
        'desktop',
        'tablet',
        'mobile',
    ]
    df = one_hot_encode(df, key, names)

    key = 'geoNetwork_country'
    logger.info(key)
    gdp = pd.read_csv('gdp_per_capita_2017.dat')
    translator = dict(zip(gdp['country'], gdp['gdp_per_capita']))
    translator[NOT_SET] = 17000
    df[key] = df[key].apply(
        lambda x: translator[x])

#     key = 'geoNetwork_continent'
#     logger.info(key)
#     names = [
#         NOT_SET,
#         'Asia',
#         'Oceania',
#         'Europe',
#         'Americas',
#         'Africa',
#     ]
#     df = one_hot_encode(df, key, names)
#  
#     key = 'geoNetwork_subContinent'
#     logger.info(key)
#     names = [
#         NOT_SET,
#         'Western Asia',
#         'Australasia',
#         'Southern Europe',
#         'Southeast Asia',
#         'Northern Europe',
#         'Southern Asia',
#         'Western Europe',
#         'South America',
#         'Eastern Asia',
#         'Eastern Europe',
#         'Northern America',
#         'Western Africa',
#         'Central America',
#         'Eastern Africa',
#         'Caribbean',
#         'Southern Africa',
#         'Northern Africa',
#         'Central Asia',
#         'Middle Africa',
#         'Melanesia',
#         'Micronesian Region',
#         'Polynesia',
#     ]
#     df = one_hot_encode(df, key, names)

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
        lambda x: float(x))

    logger.info('df.shape: %s', df.shape)
    return df


def reduce_group(group):
    reduced = []
    columns = []
    for col in group:
        columns.append(col)
        column = group[col]
        if col.startswith('totals_') or col == 'visitNumber':
            reduced.append(column.sum())
        else:
            reduced.append(column.iloc[0])
    return pd.Series(reduced, index=columns)


def groupby_session_id(df):
    logger.info('grouping by sessionId')

    gb = df.groupby('sessionId')
    logger.info('groupby size = %d', len(gb))
    df = gb.apply(reduce_group)

    logger.info('df.shape: %s', df.shape)
    return df


def load_data(data_type):
    logger.info('loading data')
    cache = data_type + '.pickle'
    if os.path.exists(cache):
        logger.info('reading cached data')
        with open(cache, 'rb') as f:
            df = pickle.load(f)
        df = df.reindex_axis(sorted(df.columns), axis=1)
        logger.info('df.shape = %s', df.shape)
        return df
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
        timezone_city, timezone_country = generate_timezones(df)
        df = drop_useless_columns(df)
        df = clean_data(df, timezone_city, timezone_country)
        df = groupby_session_id(df)
        df['totals_transactionRevenue'] = df['totals_transactionRevenue'].apply(
            lambda x: np.log(float(x) + 1))
        with open(cache, 'wb') as f:
            pickle.dump(df, f)
        df = df.reindex_axis(sorted(df.columns), axis=1)
        logger.info('df.shape = %s', df.shape)
        return df
