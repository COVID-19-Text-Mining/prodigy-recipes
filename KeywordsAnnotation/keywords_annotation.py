import random
import time
from pprint import pprint
from typing import List, Optional
import os

import prodigy
import spacy
from prodigy.components.db import get_db
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens, add_label_options
from prodigy.util import split_string

import keywords_extraction

db = get_db().get_mongo_db()
print('db.collection_names()', db.collection_names())

# global variables
# pipeline to process [{'text': '', ...}]
TEXT_STREAM_PIPELINE = []
MONGO_COL_NAME = 'entries'

# constant variables
DEFAULT_TEXT_CATEGORIES = [
    {"id": "general_info", "text": "General information"},
    {"id": "mechanism", "text": "Mechanism of cell entry"},
    {"id": "diagnostics", "text": "Diagnostics"},
    {"id": "transmission", "text": "Transmission"},
    {"id": "pathogenesis", "text": "Pathogenesis"},
    {"id": "treatment", "text": "Treatment"},
    {"id": "vaccine", "text": "Vaccine development"},
    {"id": "genomcis", "text": "Genomics"},
    {"id": "epidemiology", "text": "Epidemiological modeling/data analysis"},
    {"id": "case_report", "text": "Case report"},
    {"id": "historical", "text": "Historical information"},
]
kw_base = keywords_extraction.KeywordsExtractorBase()

############################################################################
# common functions to process stream in pipeline
############################################################################
def stream_add_options(stream, labels=None):
    """
    add options to test stream. used for text categorization tasks

    :param stream: list or iterable
    :param labels: list of str
    :return:
    """
    if labels is None:
        for task in stream:
            task["options"] = DEFAULT_TEXT_CATEGORIES
            yield task
    else:
        yield from add_label_options(stream, labels=labels)

def stream_clean_text(stream, cleaners):
    for task in stream:
        for cleaner in cleaners:
            task['text'] = cleaner(task['text'])
        yield task

def stream_add_keywords_ML(stream, kw_extractors=[], add_keywords_in_db=False):
    num_extractors = len(kw_extractors)
    for task in stream:
        keywords = []
        if 'keywords_ML' in task:
            dice = num_extractors
        else:
            dice = num_extractors-1
        if dice > 0:
            kw_extractor_index = random.randint(0, dice)
        else:
            kw_extractor_index = 0
        if kw_extractor_index < num_extractors:
            kw_extractor = kw_extractors[kw_extractor_index]
            try:
                keywords = kw_extractor.process(task['text'])
            except:
                keywords = []
            keywords_model = kw_extractor.name
        elif add_keywords_in_db:
            keywords = task.get('keywords_ML', [])
            keywords_model = 'keywords_ML'
        highlights = kw_base.keywords_to_long_highlights(
            text=task['text'],
            keywords=keywords,
        )
        prodigy_spans = kw_base.text_spans_to_token_spans(
            text_spans=highlights,
            tokens=task['tokens'],
            default_label='KEYWORD',
        )
        task['keywords'] = keywords
        task['spans'] = prodigy_spans
        task['keywords_model'] = keywords_model
        yield task


def db_endless_sampling(col_name):
    while True:
        query = db[col_name].aggregate(
            [
                {
                    '$match': {
                        'doi': {'$exists': True},
                        'abstract': {'$exists': True},
                        'title': {'$exists': True},
                    }
                },
                {'$sample': {'size': 100}},
                {
                    '$project': {
                        '_id': False,
                        'doi': True,
                        'title': True,
                        'abstract': True,
                        'keywords_ML': True,
                    },
                },
            ]
        )
        for doc in query:
            if (isinstance(doc['abstract'], str)
                and len(doc['abstract']) > 0
                and isinstance(doc['title'], str)
                and len(doc['title']) > 0
            ):
                sample = {
                    'text': doc['abstract'],
                    'title': doc['title'],
                    'meta': {'source': doc['doi']},
                }
                if ('keywords_ML' in doc
                    and isinstance(doc['keywords_ML'], list)
                    and len(doc['keywords_ML']) > 0):
                    sample['keywords_ML'] = doc['keywords_ML']
                yield sample

############################################################################
# common functions to generate html blocks for different annotation taskt
############################################################################
def get_paper_title_blocks():
    blocks = [
        {
            'view_id': 'html',
            'html_template': \
                '<center class="parent-padding-bottom-0px"><a href="https://doi.org/{{meta.source}}" '
                    'target="_blank" '
                    'style="text-decoration: none; font-size: 25px;">'
                    '{{title}}'
                '</a></center>'
        }
    ]
    return blocks

def get_task_desc_blocks(all_tasks):
    task_desc = 'Task(s): {}'.format(', '.join(all_tasks))
    blocks = [
        {
            'view_id': 'html',
            'html_template': \
                '<div class="parent-padding-top-0px parent-padding-bottom-0px" '
                    'style="text-align: right; font-size: 16px;">'
                    '{task_desc}'
                '</div>'.format(task_desc=task_desc)
        }
    ]
    return blocks

def get_ner_blocks(labels):
    blocks = [
        {
            'view_id': 'ner_manual',
            'labels': labels,
        }
    ]
    return blocks

def get_textcat_blocks(title=None, w_text=True):
    blocks = [
        {
            'view_id': 'choice',
            'label': title,
        }
    ]
    if not w_text:
        blocks[-1]['text'] = None
    return blocks

def get_summary_blocks(w_text=True):
    blocks = []
    if w_text:
        blocks.append({
            'view_id': 'text',
        })

    blocks.append(
        {
            'view_id': 'text_input',
            'field_placeholder': 'Type summary here ...',
            'field_rows': 3,
        }
    )
    return blocks

def get_note_blocks(w_text=True):
    blocks = []
    if w_text:
        blocks.append({
            'view_id': 'text',
        })

    blocks.append(
        {
            'view_id': 'text_input',
            'field_placeholder': 'Type any notes here ...',
            'field_rows': 3,
        }
    )
    return blocks

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "keywords_annotation",
    dataset_name=(
            "The dataset to use. "
            "1. Input dataset_name + dataset_file: "
            "data loaded from dataset_file, and name is dataset_name. "
            "2. Input dataset_name only: data loaded from the MongoDB "
            "collection with the same name. By default, the value is "
            "entries and load data from entries collection. ",
            "option", None, str
    ),
    dataset_file=(
            "The source data as a JSONL file",
            "option", None, str
    ),
    spacy_model=(
            "The base model",
            "option", None, str
    ),
    dataset_exclude=(
            "Names of datasets to exclude",
            "option", None, split_string
    ),
)
def COVIDKeywordsAnnotation(
    dataset_name: Optional[str] = 'entries',
    dataset_file: Optional[str] = None,
    spacy_model: Optional[str] = 'en_core_web_sm',
    dataset_exclude: Optional[List[str]] = None,
):
    """
    keywords annotation recipe

    :param dataset_name:
    :param dataset_file:
    :param spacy_model:
    :param dataset_exclude:
    :return:
    """

    # TEXT_STREAM_PIPELINE is the global variable that you put all text processors in
    # in that way, other function outside could use the same processing pipeline for the same task
    global TEXT_STREAM_PIPELINE
    # MONGO_COL_NAME is the global variable recoding which mongo collection
    # you load data if you want to load paper by doi
    global MONGO_COL_NAME

    # change globale variable MONGO_COL_NAME for further use when loading paper by doi
    MONGO_COL_NAME = dataset_name

    # Load the spaCy model for tokenization
    nlp = spacy.load(spacy_model)

    # get a text stream, which is a generator of [{'text': '', ...}]
    if dataset_file is None:
        if dataset_name in db.collection_names():
            stream = db_endless_sampling(dataset_name)
        else:
            raise ValueError(
                'Loading from database because dataset_file is not specified! '
                'However, collection {} does not exist!'.format(dataset_name)
            )
    else:
        # Load the stream from a JSONL file and return a generator that yields a
        # dictionary for each example in the data.
        stream = JSONL(dataset_file)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    # stream = add_tokens(nlp, stream)
    TEXT_STREAM_PIPELINE.append(lambda x: add_tokens(nlp, x))

    # add keywords extraction to pipeline
    kw_extractor_1 = keywords_extraction.KeywordsExtractorRaKUn(
        name='RaKUn_0',
        distance_threshold=2,
        pair_diff_length=2,
        bigram_count_threhold=2,
        num_tokens=[1, 2, 3],
        max_similar=10,
        max_occurrence=3,
        score_threshold=None,
        use_longest_phrase=True,
        ignore_shorter_keywords=False,
    )
    # use_longest_phrase = True,
    TEXT_STREAM_PIPELINE.append(
        lambda x: stream_add_keywords_ML(
            x,
            kw_extractors=[kw_extractor_1,],
            add_keywords_in_db=True,
        )
    )

    with open('keywords_annotation.html') as txt:
        template_text = txt.read()
    with open('keywords_annotation.js') as txt:
        script_text = txt.read()
    with open('keywords_annotation.css') as txt:
        css_text = txt.read()

    # activate tasks
    TASK_DESCs = {
        'ner': 'highlight named entities',
        'textcat': 'select text categories',
        'summary': 'add text summary',
        'note': 'add text notes',
    }
    AVAILABLE_TASKS = set(TASK_DESCs.keys())
    all_task_blocks = []

    # add title blocks
    all_task_blocks.extend(
        get_paper_title_blocks()
    )

    # add task desc blocks
    all_task_blocks.extend(
        get_task_desc_blocks(
            ['mark whatever you think are keywords', ])
    )

    # add keywords ner blocks
    all_task_blocks.extend(
        get_ner_blocks(labels=['KEYWORD'])
    )
    all_task_blocks.extend([
        {
            'view_id': 'html',
            'html_template': template_text,
        },
    ])

    # apply stream pipeline on text stream
    for stream_fun in TEXT_STREAM_PIPELINE:
        stream = stream_fun(stream)

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset_name,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": dataset_exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            'blocks': all_task_blocks,
            "lang": nlp.lang,
            'javascript': script_text,     # custom js
            'global_css': None,     # custom css
            'instant_submit': True,
        },
    }


###################### annotation task by doi ################################
# If it is needed to annotate by doi,
# modify prodigy_data_provider_by_doi to fit the task
# otherwise, you can comment out this part.
# Generally, nothing needs to be changed if you put all the text
# processor in TEXT_STREAM_PIPELINE.
# A text processor modifies a list of dict as [{'text': '', ...}]
# (or a generator for that) and return a generator

from prodigy_hacker import start_hacking

random_seed = random.seed(time.time())

def prodigy_data_provider_by_doi(doi):
    doc = db[MONGO_COL_NAME].find_one({'doi': doi})
    if doc and doc.get('abstract'):
        abstract = doc['abstract']

        prodigy_data = [{
            '_input_hash': hash(time.time() + random.random() * 1e6),
            '_task_hash': hash(time.time() + random.random() * 1e6),
            '_session_id': None,
            '_view_id': 'blocks',
            'text': abstract,
            'meta': {'source': doi},
        }]
        if ('keywords_ML' in doc
            and isinstance(doc['keywords_ML'], list)
            and len(doc['keywords_ML']) > 0):
            prodigy_data[0]['keywords_ML'] = doc['keywords_ML']

        stream = iter(prodigy_data)
        # apply stream pipeline on text stream
        for stream_fun in TEXT_STREAM_PIPELINE:
            stream = stream_fun(stream)

    return stream


start_hacking(prodigy_data_provider_by_doi)
