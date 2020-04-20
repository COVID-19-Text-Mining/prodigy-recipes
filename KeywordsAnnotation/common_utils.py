import importlib

def found_package(package_name):
    pkg_check = importlib.util.find_spec(package_name)
    found = pkg_check is not None
    return found


import json
from pprint import pprint
import pymongo
import regex
from datetime import datetime
import requests
import warnings

if found_package('difflib'):
    import difflib
if found_package('Levenshtein'):
    import Levenshtein


###########################################
# text comparison
###########################################
# title len stat shows that
# mean = 98.1516258677384
# std = 37.430021136179725
# percentile:
# 0.00      0
# 0.01     16
# 0.02     25
# 0.03     31
# 0.04     36
# 0.05     39
# 0.10     51
# 0.25     72
# 0.50     97
# 0.75    121
# 0.95    163
# 0.97    174
# 0.99    194
LEAST_TITLE_LEN = 16
FIVE_PERCENT_TITLE_LEN = 39
LEAST_TITLE_SIMILARITY = 0.90
IGNORE_BEGIN_END_TITLE_SIMILARITY = 0.75
# abstract len stat shows that
# mean = 1544.1819507148232
# std = 1116.3547477100615
# percentile:
# 0.00       1
# 0.01     132
# 0.02     234
# 0.03     316
# 0.04     360
# 0.05     414
# 0.10     676
# 0.25    1020
# 0.50    1394
# 0.75    1800
# 0.95    2927
# 0.97    3614
# 0.99    5909
LEAST_ABS_LEN = 132
FIVE_PERCENT_ABS_LEN = 414
LEAST_ABS_SIMILARITY = 0.90
IGNORE_BEGIN_END_ABS_SIMILARITY = 0.75

def clean_title(title):
    clean_title = title
    clean_title = clean_title.lower()
    clean_title = clean_title.strip()
    return clean_title

def text_similarity_by_char(text_1,
                            text_2,
                            quick_mode=False,
                            enable_ignore_begin_end=False,
                            ignore_begin_end_text_len=39,
                            ignore_begin_end_similarity=0.75):
    """
    calculate similarity by comparing char difference

    :param text_1:
    :param text_2:
    :return:
    """

    ref_len_ = max(float(len(text_1)), float(len(text_2)), 1.0)
    max(float(len(text_2)), 1.0)
    # find the same strings
    if not quick_mode:
        same_char = difflib.SequenceMatcher(None, text_1, text_2).get_matching_blocks()
        same_char = list(filter(lambda x: x.size > 0, same_char))
        same_char_1 = sum(
            [tmp_block.size for tmp_block in same_char]
        ) / float(max(len(text_1), len(text_2), 1.0))
        same_char_2 = 0
        if enable_ignore_begin_end and len(same_char) > 0:
            text_1 = text_1[same_char[0].a: same_char[-1].a + same_char[-1].size]
            text_2 = text_2[same_char[0].b: same_char[-1].b + same_char[-1].size]
            if (len(text_1) > ignore_begin_end_text_len
                and len(text_2) > ignore_begin_end_text_len
                and len(text_1)/max(len(text_1), 1.0) > ignore_begin_end_similarity
                and len(text_2)/max(len(text_2), 1.0) > ignore_begin_end_similarity
            ):
                same_char_2 = sum(
                    [tmp_block.size for tmp_block in same_char]
                ) / float(max(len(text_1), len(text_2), 1.0))
        same_char_ratio = max(same_char_1, same_char_2)
    # find the different strings
    diff_char_ratio = 1 - Levenshtein.distance(text_1, text_2) / float(
        max(min(len(text_1), len(text_2)), 1.0))
    diff_char_ratio = max(diff_char_ratio, 0.0)

    if not quick_mode:
        similarity = (same_char_ratio + diff_char_ratio) / 2.0
    else:
        similarity = diff_char_ratio
    # print(text_1, text_2, same_char, diff_char, similarity, maxlarity, answer_simis)
    return similarity

###########################################
# parse web data
###########################################
PATTERN_DATE_0 =[
    '%m/%d/%Y',
    '%Y-%m-%d',
    '%Y',
    '%Y %b %d',
    '%Y %b',
]

MONTH_DICT = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
}
SEASON_DICT = {
    'spring': 1,
    'summer': 4,
    'autumn': 7,
    'fall': 7,
    'winter': 10,
}

PATTERN_DATE_1 = [
    regex.compile(
        '(?P<year>[0-9]{{4}}) +(?P<month>{}).*'.format(
            '|'.join(list(MONTH_DICT.keys()))
        )
    ),
    regex.compile(
        '(?P<year>[0-9]{{4}}) +(?P<season>{}).*'.format(
            '|'.join(list(SEASON_DICT.keys()))
        )
    ),
    regex.compile(
        '.*(?P<year>[0-9]{4})-(?P<month>[0-9]{1,2})-(?P<day>[0-9]{1,2}).*'
    ),
]

def parse_date(date_obj):
    date = {}
    if isinstance(date_obj, str):
        date = parse_date_str(date_obj)
    if isinstance(date_obj, list):
        date = parse_date_list(date_obj)
    if len(date) < 3:
        for k in ({'year', 'month', 'day'} - set(date.keys())):
            date[k] = None
    return date

def parse_date_list(date_list):
    time_parsed = {}
    if len(date_list) == 3 and len(date_list[0])==4:
        time_parsed = {
            'year': int(date_list[0]),
            'month': int(date_list[1]),
            'day': int(date_list[2]),
        }
    if len(date_list) == 2 and len(date_list[0])==4:
        time_parsed = {
            'year': int(date_list[0]),
            'month': int(date_list[1]),
        }
    return time_parsed

def parse_date_str(date_str):
    time_parsed = {}
    date_str = date_str.strip()
    for a_pattern in PATTERN_DATE_0:
        try:
            result = datetime.strptime(date_str, a_pattern)
            if '%Y' in a_pattern:
                time_parsed['year'] = result.year
            if '%m' in a_pattern or '%b' in a_pattern:
                time_parsed['month'] = result.month
            if '%d' in a_pattern:
                time_parsed['day'] = result.day
            break
        except:
            pass
    if len(time_parsed) == 0:
        for a_pattern in PATTERN_DATE_1:
            tmp_m = a_pattern.match(date_str.lower())
            if tmp_m:
                result = tmp_m.groupdict()
                if 'year' in result:
                    time_parsed['year'] = int(result['year'])
                if 'month' in result:
                    if result['month'] in MONTH_DICT:
                        time_parsed['month'] = int(MONTH_DICT[result['month']])
                    else:
                        time_parsed['month'] = int(result['month'])
                if 'season' in result:
                    time_parsed['month'] = int(SEASON_DICT[result['season']])
                if 'day' in result:
                    time_parsed['day'] = int(result['day'])
                break
    return time_parsed

def parse_names(name_obj):
    names = []
    if isinstance(name_obj, str):
        names = parse_names_str(name_obj)
    if isinstance(name_obj, list):
        names = parse_names_list(name_obj)
    return names

def parse_names_str(name_str):
    names = []
    fragments = []
    name_str = name_str.strip()

    if ';' in name_str:
        fragments = name_str.split(';')
    elif (name_str.count(',') > 1
         or (name_str.count(',')==1 and name_str.count(' ')>2)
    ):
        fragments = name_str.split(',')
    elif (name_str.count(',')==1 and name_str.count(' ')<=2):
        fragments = [name_str]
    elif ' ' in name_str:
        fragments = [name_str]
    for frag in fragments:
        tmp_name = None
        if ',' in frag:
            pieces = frag.split(',')
            tmp_name = {
                'last': pieces[0].strip(),
                'first': pieces[-1].strip(),
            }
        elif ' ' in frag:
            pieces = frag.split(' ')
            tmp_name = {
                'last': pieces[-1].strip(),
                'first': pieces[0].strip(),
            }
        else:
            tmp_name = {
                'first': frag.strip(),
                'last': None
            }
        if tmp_name is not None:
            names.append(tmp_name)
    return names

def parse_names_list(name_list):
    names = []
    for n in name_list:
        names.append({
            'first': n.get('given', None),
            'last': n.get('family', None),
        })
    return names

###########################################
# communicate with doi.org
###########################################

def query_doiorg_by_doi(doi):
    # goal
    query_results = None
    # query doi org
    query_url = 'http://dx.doi.org/{}'.format(doi)
    headers = {"accept": "application/x-bibtex"}

    try:
        query_results = requests.get(
            query_url,
            headers=headers
        )
    except:
        raise ConnectionError(
            'Request to doi.org failed when searching doi: {}!'.format(doi)
        )
    if query_results.reason != 'OK':
        raise ValueError(
            'Error result from doi.org when searching doi: {}!'.format(doi)
        )

    return query_results


###########################################
# communicate with crossref
###########################################

def query_crossref_by_doi(doi, verbose=True):
    # goal
    crossref_results = None

    # query crossref
    query_url = 'https://api.crossref.org/works/{}'.format(doi)
    try:
        query_results = requests.get(
            query_url,
        )
    except:
        raise ConnectionError(
            'Request to crossref failed when searching doi: {}!'.format(doi)
        )
    try:
        query_results = query_results.json()
    except Exception as e:
        raise ValueError(
            'Query result from crossref cannot be jsonified when searching doi: {}!'.format(doi)
        )

    # filter out empty query results
    if ('message' in query_results
        and isinstance(query_results['message'], dict)
    ):
        crossref_results = query_results['message']
    else:
        warnings.warn(
            'Empty result from crossref when searching doi: {}'.format(doi)
        )

    return crossref_results


def query_crossref(query_params):
    # goal
    crossref_results = None

    # query crossref
    query_url = 'https://api.crossref.org/works'
    try:
        query_results = requests.get(
            query_url,
            params=query_params,
        )
    except:
        raise ConnectionError(
            'Request to crossref failed when querying by: {}!'.format(query_params)
        )
    try:
        query_results = query_results.json()
    except Exception as e:
        raise ValueError(
            'Query result from crossref cannot be jsonified when querying by: {}!'.format(query_params)
        )

    # filter out empty query results
    if ('message' in query_results
        and 'items' in query_results['message']
        and isinstance(query_results['message']['items'], list)
        and len(query_results['message']['items']) > 0
    ):
        crossref_results = query_results['message']['items']
    else:
        warnings.warn(
            'Empty result from crossref when querying by: {}!'.format(query_params)
        )

    return crossref_results


def valid_a_doi(doi, doc_data=None, abstract=None, title=None):
    valid = True

    # type check
    if valid:
        if not (isinstance(doi, str) and len(doi) > 0):
            valid = False
            print('DOI should be a non-empty str. doi: {} might be invalid'.format(doi))

    # check via crossref
    if valid:
        try:
            query_result = query_crossref_by_doi(doi)
        except Exception as e:
            query_result = None
            valid = False
            print(e)
        if (query_result is not None
            and 'abstract' in query_result
            and len(query_result['abstract']) > 0
            and isinstance(abstract, str)
            and len(abstract) > 0
        ):
            similarity = text_similarity_by_char(
                query_result['abstract'],
                abstract,
                quick_mode=False,
                enable_ignore_begin_end=True,
                ignore_begin_end_text_len=FIVE_PERCENT_ABS_LEN,
                ignore_begin_end_similarity=IGNORE_BEGIN_END_ABS_SIMILARITY,
            )
            if (len(query_result['abstract']) > LEAST_ABS_LEN
                and len(abstract) > LEAST_ABS_LEN
                and similarity < 0.6
            ):
                valid = False
                print('Abstract does not match. doi: {} might be invalid!'.format(doi))
                print('abstract similarity',  similarity)

        if (query_result is not None
            and 'title' in query_result
            and isinstance(query_result['title'], list)
            and len(query_result['title']) > 0
            and isinstance(title, str)
            and len(title) > 0
        ):
            similarity = text_similarity_by_char(
                clean_title(query_result['title'][0]),
                clean_title(title),
                quick_mode=False,
                enable_ignore_begin_end=True,
                ignore_begin_end_text_len=FIVE_PERCENT_TITLE_LEN,
                ignore_begin_end_similarity=IGNORE_BEGIN_END_TITLE_SIMILARITY,
            )
            if (len(query_result['title'][0]) > LEAST_TITLE_LEN
                and len(title) > LEAST_TITLE_LEN
                and similarity < 0.6
            ):
                valid = False
                print('Title does not match. doi: {} might be invalid!'.format(doi))
                print('Title similarity',  similarity)

    if valid:
        # check via doi.org
        try:
            query_result = query_doiorg_by_doi(doi)
        except Exception as e:
            query_result = None
            valid = False
            print(e)

    return valid

###########################################
# communicate with mongodb
###########################################

def get_mongo_db(config_file_path):
    """
    read config file and return mongo db object

    :param config_file_path: path to config file. config is a dict of dict as
                config = {
                'mongo_db': {
                    'host': 'mongodb05.nersc.gov',
                    'port': 27017,
                    'db_name': 'COVID-19-text-mining',
                    'username': '',
                    'password': '',
                }
            }
    :return:
    """
    with open(config_file_path, 'r') as fr:
        config = json.load(fr)

    client = pymongo.MongoClient(
        host=config['mongo_db']['host'],
        port=config['mongo_db']['port'],
        username=config['mongo_db']['username'],
        password=config['mongo_db']['password'],
        authSource=config['mongo_db']['db_name'],
    )
    db = client[config['mongo_db']['db_name']]
    return db