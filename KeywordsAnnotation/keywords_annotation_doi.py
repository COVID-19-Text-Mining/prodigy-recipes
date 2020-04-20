import os
import sys

current_folder = os.path.abspath(
    os.path.dirname(__file__),
)
print('current_folder', current_folder)
if current_folder not in sys.path:
    sys.path.append(current_folder)

import json
from typing import List, Optional
from urllib.parse import urlparse, parse_qs
import time
import random

import prodigy.app
import spacy
from prodigy.app import app as prodigy_app
from prodigy.components.loaders import JSONL
from prodigy.components.preprocess import add_tokens
from prodigy.util import split_string
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import request_response

import keywords_extraction
from common_utils import get_mongo_db

with open('keywords_annotation.html') as txt:
    template_text = txt.read()
with open('keywords_annotation.js') as txt:
    script_text = txt.read()
with open('keywords_annotation_doi.css') as txt:
    css_text = txt.read()

random_seed = random.seed(time.time())
tokenizer_spacy = spacy.load('en_core_web_sm')
kw_base = keywords_extraction.KeywordsExtractorBase()
kw_extractor = keywords_extraction.KeywordsExtractorRaKUn(
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
db = get_mongo_db('tmp_db_config.json')
print('db.collection_names()', db.collection_names())

# Recipe decorator with argument annotations: (description, argument type,
# shortcut, type / converter function called on value before it's passed to
# the function). Descriptions are also shown when typing --help.
@prodigy.recipe(
    "keywords_annotation_doi",
    dataset=("The dataset to use", "positional", None, str),
    spacy_model=("The base model", "positional", None, str),
    source=("The source data as a JSONL file", "positional", None, str),
    label=("One or more comma-separated labels", "option", "l", split_string),
    exclude=("Names of datasets to exclude", "option", "e", split_string),
)
def keywords_annotation(
        dataset: str,
        spacy_model: str,
        source: str,
        label: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
):
    """
    Mark spans manually by token. Requires only a tokenizer and no entity
    recognizer, and doesn't do any active learning.
    """
    # Load the spaCy model for tokenization
    nlp = spacy.load(spacy_model)

    # Load the stream from a JSONL file and return a generator that yields a
    # dictionary for each example in the data.
    stream = JSONL(source)

    # Tokenize the incoming examples and add a "tokens" property to each
    # example. Also handles pre-defined selected spans. Tokenization allows
    # faster highlighting, because the selection can "snap" to token boundaries.
    stream = add_tokens(nlp, stream)

    return {
        "view_id": "blocks",  # Annotation interface to use
        "dataset": dataset,  # Name of dataset to save annotations
        "stream": stream,  # Incoming stream of examples
        "exclude": exclude,  # List of dataset names to exclude
        "config": {  # Additional config settings, mostly for app UI
            'blocks': [
                {'view_id': 'ner_manual'},
                {
                    'view_id': 'html',
                    'html_template': template_text,
                },
            ],
            "lang": nlp.lang,
            "labels": label,  # Selectable label options
            'javascript': script_text,
            'global_css': css_text,
            'instant_submit': True,
        },
    }


def get_prodigy_json_data_by_doi(doi):
    prodigy_data = {
        '_input_hash': hash(time.time()+random.random()*1e6),
        '_task_hash': hash(time.time()+random.random()*1e6),
        '_session_id': None,
        '_view_id': 'blocks',
        'text': '',
        'keywords': [],
        'spans': [],
        'meta': {'source': doi},
        'tokens': [],
    }

    abstract = None
    keywords = []
    doc = db['entries'].find_one({'doi': doi})
    if doc and doc.get('abstract'):
        abstract = doc['abstract']
        abstract = kw_base.clean_html_tag(abstract)
        spacy_doc = tokenizer_spacy(abstract)
        tokens = kw_base.reformat_tokens(spacy_doc)
        try:
            keywords = kw_extractor.process(abstract)
        except:
            print('Error')
            print(abstract)
        highlights = kw_base.keywords_to_long_highlights(
            text=abstract,
            keywords = keywords,
        )
        prodigy_spans = kw_base.text_spans_to_token_spans(
            text_spans=highlights,
            tokens=tokens,
            default_label='KEYWORD',
        )
        prodigy_data['text'] = abstract
        prodigy_data['keywords'] = keywords
        prodigy_data['spans'] = prodigy_spans
        prodigy_data['tokens'] = tokens

    return prodigy_data


def hacky_get_questions(request: Request):
    referer = request.headers.get("referer")
    query = parse_qs(urlparse(referer).query)
    doi = query.get('doi', None)
    if isinstance(doi, list) and len(doi) == 1:
        doi = doi[0]
    elif not isinstance(doi, str):
        doi = 'No doi!'
    print('doi', doi)
    # TODO: deal with no doi situation
    prodigy_data = get_prodigy_json_data_by_doi(doi)

    result = {
        'total': 1,
        'progress': None,
        'session_id': None,
        'tasks': [prodigy_data]
    }

    return JSONResponse(result)


for i, route in enumerate(prodigy_app.router.routes):
    if route.path == '/get_session_questions':
        route.endpoint = hacky_get_questions
        route.app = request_response(hacky_get_questions)
        break
