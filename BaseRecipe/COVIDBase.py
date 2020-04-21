import os
import sys

current_folder = os.path.abspath(
    os.path.dirname(__file__),
)
print('current_folder', current_folder)
if current_folder not in sys.path:
    sys.path.append(current_folder)


import time
import random
import prodigy
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens, add_label_options
from prodigy.util import split_string
import spacy
from typing import List, Optional

# with open('keywords_annotation.html') as txt:
#     template_text = txt.read()
# with open('keywords_annotation.js') as txt:
#     script_text = txt.read()
# with open('keywords_annotation.css') as txt:
#     css_text = txt.read()

from common_utils import get_mongo_db
try:
    db = get_mongo_db('tmp_db_config.json')
    print('db.collection_names()', db.collection_names())
except:
    db = None

# pipeline to process [{'text': '', ...}]
TEXT_STREAM_PIPELINE = []

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
            }
        )

    blocks.append(
        {
            'view_id': 'text_input',
            'field_placeholder': 'Type summary here ...',
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
        "Available values: NER, TextCat, Summary",
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
    task_type: Optional[List[str]] = ['NER', 'TextCat', 'Summary'],
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
    Mark spans manually by token. Requires only a tokenizer and no entity
    recognizer, and doesn't do any active learning.
    """

    global TEXT_STREAM_PIPELINE

    # Load the spaCy model for tokenization
    nlp = spacy.load(spacy_model)

    # get a text stream, which is a generator of [{'text': '', ...}]
    if dataset_file is None:
        # TODO: implement this after implementing custom loader
        # load from collection dataset_name
        ...
    else:
        # Load the stream from a JSONL file and return a generator that yields a
        # dictionary for each example in the data.
        stream = JSONL(dataset_file)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    # stream = add_tokens(nlp, stream)
    TEXT_STREAM_PIPELINE.append(lambda x: add_tokens(nlp, x))

    # activate tasks
    AVAILABLE_TASKS = ['ner', 'textcat', 'summary']
    task_type = set([x.lower() for x in task_type])
    all_task_blocks = []
    text_showed = False
    if len(set(task_type) - set(AVAILABLE_TASKS)) > 0:
        raise ValueError(
            'task_type {} not enabled. Available task types: {}'.format(
                task_type,
                AVAILABLE_TASKS
            ))


    if 'ner' in task_type:
        all_task_blocks.extend(
            get_ner_blocks(labels=ner_label)
        )
        text_showed = True
        textcat_title = None

    if 'textcat' in task_type:
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
        all_task_blocks.extend(
            get_summary_blocks(
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
            'javascript': None,     # custom js
            'global_css': None,     # custom css
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
tokenizer_spacy = spacy.load('en_core_web_sm')



def prodigy_data_provider_by_doi(doi):
    global TEXT_STREAM_PIPELINE

    doc = db['entries'].find_one({'doi': doi})
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

        stream = iter(prodigy_data)
        # apply stream pipeline on text stream
        for stream_fun in TEXT_STREAM_PIPELINE:
            stream = stream_fun(stream)

    return stream

# start_hacking(prodigy_data_provider_by_doi)