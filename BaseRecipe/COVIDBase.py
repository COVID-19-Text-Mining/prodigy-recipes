import random
import time
from typing import List, Optional

import prodigy
import spacy
from prodigy.components.db import get_db
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens, add_label_options
from prodigy.util import split_string

# # for debugging locally
# # any code is ok here as long as it provides a mongo db obj
# import os
# import sys
# parent_folder = os.path.abspath(os.path.join(
#     os.path.dirname(__file__),
#     '..'
# ))
# print('parent_folder', parent_folder)
# if parent_folder not in sys.path:
#     sys.path.append(parent_folder)
# from common_utils import get_mongo_db
# db = get_mongo_db('../config.json')

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
                yield {
                    'text': doc['abstract'],
                    'title': doc['title'],
                    'meta': {'source': doc['doi']},
                }

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
    "COVIDBase",
    task_type=(
            "One or more comma-separated task types. "
            "Available values: NER, TextCat, Summary, Note",
            "option", None, split_string
    ),
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
    ner_label=(
            "One or more comma-separated labels",
            "option", None, split_string
    ),
    textcat_title=(
            "Description for labels in text categorization task",
            "option", None, str
    ),
    textcat_label=(
            "One or more comma-separated labels",
            "option", None, split_string
    ),
    disable_multiple_choice=(
            "If passed to command line, disable multiple choice for tasks "
            "using choice components (such as text categorization)",
            "flag", None, bool
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
def COVIDBase(
        task_type: Optional[List[str]] = ['NER', 'TextCat', 'Summary', 'Note'],
        dataset_name: Optional[str] = 'entries',
        dataset_file: Optional[str] = None,
        ner_label: Optional[List[str]] = None,
        textcat_title: Optional[str] = None,
        textcat_label: Optional[List[str]] = None,
        disable_multiple_choice: bool = False,
        spacy_model: Optional[str] = 'en_core_web_sm',
        dataset_exclude: Optional[List[str]] = None,
):
    """
    The base recipe provides most common annotation tasks for COVID study.
    Besides, you can build more sophisticated recipe based on this and
    save some time building wheels.
    Currently, there are four default tasks you can choose among.
    You can specify `task-type` to select.
    The default value is `-task-type ner, textcat, summary, note`.
    1. NER (Named-Entity Recognition): mark the words or phrase in text with different labels.
        You can define the labels with `ner-label. E.g. `-ner-label vaccine,disease`
    2. TextCat (Text categorization): choose the categories the paragraph falls in.
        Multiple choice is enabled by default.
        You can define the labels with `textcat_label`. E.g. `-textcat-lebel mechanism,diagnostics`.
        The default value is eleven classes defined by our Expert, Kevin.
    3. Summary: Summary the paragraph with more consice sentences.
    4. Note: Add any note you like, such as summary, important points, critical parameters, etc.

    :param task_type:
    :param dataset_name:
    :param dataset_file:
    :param ner_label:
    :param textcat_title:
    :param textcat_label:
    :param disable_multiple_choice:
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

    # custom html, javascript, css
    # If you want to design your own html, javascript, and css
    # load them from file as following and pass them to corresponding interface
    # In terms of html, you can create a block with `view_id` of `html`
    # and pass the `template_text` to `html_template`.
    # Then, add this block to all_task_blocks.
    # E.g:
    # all_task_blocks.append({
    #     'view_id': 'html',
    #     'html_template': template_text,
    # })
    # In terms of javascript, pass `script_text` to `javascript` when returning.
    # If you are familiar with React, you can modify bundle.js as a more flexible way.
    # E.g.:
    # return {
    #     ...
    #     "config": {  # Additional config settings, mostly for app UI
    #         ...
    #         'javascript': script_text,  # custom js
    #         ...
    #     },
    #     ...
    # }
    # In terms of css, it is a little bit tricky. There are two ways.
    # (1) add the css in index.html.
    # E.g.:
    # <head>
    #     ...
    #     <style>
    #         .prodigy-content {
    #             text-align: justify !important;
    #         }
    #     </style>
    #     ...
    # </head>
    # (2) Pass css_text to `global_css` when returning. However, this is slower than method (1).
    # E.g.:
    # return {
    #     ...
    #     "config": {  # Additional config settings, mostly for app UI
    #         ...
    #         'css_text': script_text,  # custom js
    #         ...
    #     },
    #     ...
    # }
    # with open('keywords_annotation.html') as txt:
    #     template_text = txt.read()
    # with open('keywords_annotation.js') as txt:
    #     script_text = txt.read()
    # with open('keywords_annotation.css') as txt:
    #     css_text = txt.read()

    # activate tasks
    TASK_DESCs = {
        'ner': 'highlight named entities',
        'textcat': 'select text categories',
        'summary': 'add text summary',
        'note': 'add text notes',
    }
    AVAILABLE_TASKS = set(TASK_DESCs.keys())
    task_type = [x.lower() for x in task_type]
    all_task_blocks = []
    text_showed = False
    if len(set(task_type) - AVAILABLE_TASKS) > 0:
        raise ValueError(
            'task_type {} not enabled. Available task types: {}'.format(
                task_type,
                AVAILABLE_TASKS
            ))

    # add title blocks
    all_task_blocks.extend(
        get_paper_title_blocks()
    )

    # add task desc blocks
    all_task_blocks.extend(
        get_task_desc_blocks(
            [TASK_DESCs[t] for t in task_type])
    )

    if 'ner' in task_type:
        # add ner blocks
        all_task_blocks.extend(
            get_ner_blocks(labels=ner_label)
        )
        text_showed = True
        textcat_title = None

    if 'textcat' in task_type:
        # add text categorization blocks
        TEXT_STREAM_PIPELINE.append(
            lambda x: stream_add_options(x, labels=textcat_label)
        )
        all_task_blocks.extend(
            get_textcat_blocks(
                title=textcat_title,
                w_text=(not text_showed)
            )
        )
        text_showed = True

    if 'summary' in task_type:
        # add summary blocks
        all_task_blocks.extend(
            get_summary_blocks(
                w_text=(not text_showed)
            )
        )
        text_showed = True

    if 'note' in task_type:
        # add note blocks
        all_task_blocks.extend(
            get_note_blocks(
                w_text=(not text_showed)
            )
        )
        text_showed = True

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
            'javascript': None,  # custom js
            'global_css': None,  # custom css
            'instant_submit': True,
            'choice_style': 'single' if disable_multiple_choice else 'multiple',
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
        # get data from db
        abstract = doc['abstract']
        prodigy_data = [{
            '_input_hash': hash(time.time() + random.random() * 1e6),
            '_task_hash': hash(time.time() + random.random() * 1e6),
            '_session_id': None,
            '_view_id': 'blocks',
            'text': abstract,
            'meta': {'source': doi},
        }]

        # process data into the format fitting the task
        stream = iter(prodigy_data)
        # apply stream pipeline on text stream
        for stream_fun in TEXT_STREAM_PIPELINE:
            stream = stream_fun(stream)

    return stream


start_hacking(prodigy_data_provider_by_doi)
