import sys
import os
import importlib

def found_package(package_name):
    pkg_check = importlib.util.find_spec(package_name)
    found = pkg_check is not None
    return found


import collections
import json
from pprint import pprint
import numpy as np
from bson import json_util
import re
import regex
import string
import copy
import spacy
from nltk.corpus import stopwords
from typing import List, Union

if found_package('matplotlib'):
    import matplotlib.pyplot as plt

if found_package('wordcloud'):
    from wordcloud import WordCloud

if found_package('summa'):
    import summa
if found_package('yake'):
    import yake
if found_package('pke'):
    import pke
if found_package('mrakun'):
    import mrakun

if os.path.exists('./NNSupport'):
    sys.path.append(os.path.abspath("./NNSupport"))
    from onmt.translate.translator import build_translator
    from onmt.keyphrase.utils import meng17_tokenize, replace_numbers_to_DIGIT
    from strsimpy import NGram


class KeywordsExtractorBase():
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'Base')
        # output_format could be words_only or words_and_scores
        self.output_format = kwargs.get('output_format', 'words_only')
        self.score_threshold = kwargs.get('score_threshold', None)
        self.use_longest_phrase = kwargs.get('use_longest_phrase', False)
        self.only_extractive = kwargs.get('only_extractive', True)
        self.ignore_shorter_keywords = kwargs.get('ignore_shorter_keywords', True)

    def process(self, text):
        words_and_scores = self._process(text)
        if self.score_threshold:
            words_and_scores = list(filter(
                lambda x: x[1] > self.score_threshold,
                words_and_scores
            ))
        if self.only_extractive:
            words_and_scores = list(filter(
                lambda x: regex.search(
                    r'\b{}\b'.format(regex.escape(x[0], special_only=True)),
                    text,
                    flags=regex.IGNORECASE
                ),
                words_and_scores
            ))
        if self.output_format == 'words_and_scores':
            return words_and_scores
        elif self.output_format == 'words_only':
            keywords = list(set([item[0] for item in words_and_scores]))
            if self.use_longest_phrase:
                keywords, _ = self.get_longest_phrase(
                    text=text,
                    keywords=keywords,
                    ignore_shorter=self.ignore_shorter_keywords,
                )
            return keywords
        else:
            raise AttributeError(
                'output_format {} not recognized'.format(self.output_format)
            )

    def _process(self, text):
        raise NotImplementedError

    def hightlight_keywords(self,
                            text,
                            keywords,
                            light_color='#ffea593d',
                            deep_color='#ffc107'):
        hightlighted_html = ''
        all_hightlights = []
        tokens = self.full_tokenize(text)
        for t in tokens:
            t['background_color'] = []

        for w in keywords:
            matches = regex.finditer(
                r'\b{}\b'.format(regex.escape(w, special_only=True)),
                text,
                flags=regex.IGNORECASE
            )
            all_hightlights.extend([
                {
                    'start': m.start(),
                    'end': m.end(),
                    'text': m.group(),
                }
                for m in matches
            ])

        all_hightlights = sorted(all_hightlights, key=lambda x: x['start'])

        for h in all_hightlights:
            for t in tokens:
                if (t['start'] >= h['start'] and t['end'] <= h['end']):
                    t['background_color'].append(light_color)

        for t in tokens:
            color_len = len(t['background_color'])
            if color_len == 0:
                hightlighted_html += t['text']
            elif color_len == 1:
                hightlighted_html += \
                    '<span style="background-color:{background_color};">{text}</span>'.format(
                    background_color=t['background_color'][0],
                    text=t['text']
                )
            else:
                hightlighted_html += \
                    '<span style="background-color:{background_color};">{text}</span>'.format(
                    background_color=deep_color,
                    text=t['text']
                )
        return hightlighted_html

    def keywords_to_highlights(self, text, keywords):
        # get all matched in original text
        all_hightlights = []
        for w in keywords:
            matches = regex.finditer(
                r'\b{}\b'.format(regex.escape(w, special_only=True)),
                text,
                flags=regex.IGNORECASE
            )
            all_hightlights.extend([
                {
                    'start': m.start(),
                    'end': m.end(),
                    'text': m.group(),
                }
                for m in matches
            ])

        all_hightlights = sorted(all_hightlights, key=lambda x: x['start'])
        return all_hightlights

    def keywords_to_long_highlights(self, text, keywords):
        # get all matched in original text
        all_hightlights = self.keywords_to_highlights(text=text, keywords=keywords)

        # connect adjecent keywords
        long_highlights = []
        for h in all_hightlights:
            if len(long_highlights) == 0:
                long_highlights.append(h)
                continue
            if text[long_highlights[-1]['end']: h['start']].strip() == '':
                # merge
                long_highlights[-1]['end'] = max(long_highlights[-1]['end'], h['end'])
                long_highlights[-1]['text'] = text[
                   long_highlights[-1]['start']: long_highlights[-1]['end']
                ]
            else:
                long_highlights.append(h)
        return long_highlights

    def highlights_to_keywords(self, long_highlights, ignore_shorter=True):
        # ignore repeated but shorter keywords
        long_highlights = sorted(long_highlights, key=lambda x: len(x['text']), reverse=True)
        final_keywords = []
        final_highlights = []
        for h in long_highlights:
            to_add = True
            for k in final_keywords:
                if ignore_shorter:
                    if regex.search(
                            r'\b{}\b'.format(regex.escape(h['text'], special_only=True)),
                            k['text'],
                            flags=regex.IGNORECASE
                    ):
                        to_add = False
                        break
                else:
                    if h['text'].lower() == k['text'].lower():
                        to_add = False
                        break
            if to_add:
                final_keywords.append(h)
        final_keywords = [x['text'] for x in final_keywords]
        final_keywords_set_lower = set([x.lower() for x in final_keywords])
        final_highlights = list(filter(
            lambda h: h['text'].lower() in final_keywords_set_lower, long_highlights
        ))
        final_highlights = sorted(final_highlights, key=lambda x: x['start'])
        return final_keywords, final_highlights

    def get_longest_phrase(self, text, keywords, ignore_shorter=True):
        long_highlights = self.keywords_to_long_highlights(text=text, keywords=keywords)
        final_keywords, final_highlights = self.highlights_to_keywords(
            long_highlights=long_highlights,
            ignore_shorter=ignore_shorter
        )
        return final_keywords, final_highlights

    def reformat_tokens(self, tokens):
        reformated_tokens = []
        for i, t in enumerate(tokens):
            if not isinstance(t, dict):
                if 'start' in dir(t) and 'end' in dir(t) and 'text' in t:
                    # chemdataextractor token format
                    reformated_tokens.append({
                        'start': t.start,
                        'end': t.end,
                        'text': t.text,
                    })
                elif 'idx' in dir(t) and 'text' in dir(t):
                    # spacy token format
                    reformated_tokens.append({
                        'start': t.idx,
                        'end': t.idx+len(t.text),
                        'text': t.text,
                    })
            else:
                reformated_tokens.append({
                    'start': t['start'],
                    'end': t['end'],
                    'text': t['text'],
                })
            reformated_tokens[-1]['id'] = i
        return reformated_tokens

    def _reformat_text_spans(self, text_spans, default_label):
        reformated_spans = copy.deepcopy(text_spans)
        for s in reformated_spans:
            if 'label' not in s:
                s['label'] = default_label
        return reformated_spans

    def text_spans_to_token_spans(self,
                                  text_spans,
                                  tokens,
                                  default_label='KEYWORD'):
        token_spans = []

        text_spans = self._reformat_text_spans(
            text_spans,
            default_label=default_label
        )
        tokens = self.reformat_tokens(tokens)
        for s in text_spans:
            text_tokens = list(filter(
                lambda t: t['start'] >= s['start'] and t['end'] <= s['end'],
                tokens
            ))
            if (len(text_tokens) > 0
                and text_tokens[0]['start'] == s['start']
                and text_tokens[-1]['end'] == s['end']):
                token_spans.append({
                    'start': s['start'],
                    'end': s['end'],
                    'label': s['label'],
                    'token_start': text_tokens[0]['id'],
                    'token_end': text_tokens[-1]['id'],
                })
        return token_spans

    def full_tokenize(self, text):
        tokens = []
        for m in regex.finditer(r'\p{Punct}|.+?\b', text):
            tokens.append({
                'start': m.start(),
                'end': m.end(),
                'text': m.group(),
            })
        return tokens

    def clean_html_tag(self, text):
        new_text = regex.sub(r'<.{1,10}?>', '', text)
        return new_text


class KeywordsExtractorSumma(KeywordsExtractorBase):
    def __init__(self, split=False, scores=False, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'Summa')
        self.split=split
        self.scores = scores

    def _process(self, text):
        keywords = summa.keywords.keywords(text, split=self.split, scores=self.scores)
        return keywords


class KeywordsExtractorYake(KeywordsExtractorBase):
    def __init__(self, max_ngram_size=3, window_size=1, top=20, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'Yake')
        self.max_ngram_size = max_ngram_size
        self.window_size = window_size
        self.top = top
        self.kw_extractor = yake.KeywordExtractor(
            n=self.max_ngram_size,
            windowsSize=self.window_size,
            top=self.top,
        )

    def _process(self, text):
        keywords = self.kw_extractor.extract_keywords(text)
        return keywords


class KeywordsExtractorTfIdf(KeywordsExtractorBase):
    def __init__(self, max_ngram_size=3, df=None, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'TfIdf')
        self.max_ngram_size = max_ngram_size
        self.df = df
        if isinstance(self.df, str):
            self.df = pke.load_document_frequency_file(input_file=self.df)
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        self.kw_extractor = pke.unsupervised.TfIdf()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            n=self.max_ngram_size,
            stoplist=self.stoplist,
        )

        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        self.kw_extractor.candidate_weighting(df=self.df)

        # get the 10-highest scored candidates as keyphrases
        num_keywords = len(self.kw_extractor.candidates)
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        else:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)

        return keywords


class KeywordsExtractorKPMiner(KeywordsExtractorBase):
    def __init__(self, lasf=3, cutoff=200, alpha=2.3, sigma=3.0, df=None, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'KPMiner')
        self.lasf = lasf
        self.cutoff = cutoff
        self.alpha = alpha
        self.sigma = sigma
        self.df = df
        if isinstance(self.df, str):
            self.df = pke.load_document_frequency_file(input_file=self.df)
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        self.kw_extractor = pke.unsupervised.KPMiner()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            lasf=self.lasf,
            cutoff=self.cutoff,
            stoplist=self.stoplist,
        )

        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        self.kw_extractor.candidate_weighting(
            alpha=self.alpha,
            sigma=self.sigma,
            df=self.df,
        )

        # get the 10-highest scored candidates as keyphrases
        num_keywords = len(self.kw_extractor.candidates)
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        else:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)

        return keywords


class KeywordsExtractorSingleRank(KeywordsExtractorBase):
    def __init__(self, window=10, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'SingleRank')
        self.window = window
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        self.kw_extractor = pke.unsupervised.SingleRank()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            pos=self.pos,
        )

        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        self.kw_extractor.candidate_weighting(
            window=self.window,
            pos=self.pos,
        )

        # get the 10-highest scored candidates as keyphrases
        num_keywords = len(self.kw_extractor.candidates)
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        else:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)

        return keywords


class KeywordsExtractorTopicRank(KeywordsExtractorBase):
    def __init__(self, threshold=0.74, method='average', heuristic=None, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'TopicRank')
        self.threshold = threshold
        self.method = method
        self.heuristic = heuristic
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        self.kw_extractor = pke.unsupervised.TopicRank()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            pos=self.pos,
            stoplist=self.stoplist,
        )
        num_keywords = len(self.kw_extractor.candidates)

        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        if num_keywords > 0:
            try:
                self.kw_extractor.candidate_weighting(
                    threshold=self.threshold,
                    method=self.method,
                    heuristic=self.heuristic,
                )
            except:
                num_keywords = 0

        # get the 10-highest scored candidates as keyphrases
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        elif num_keywords > 0:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)
        else:
            keywords = []

        return keywords


class KeywordsExtractorTopicalPageRank(KeywordsExtractorBase):
    def __init__(self, window=10, lda_model=None, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'TopicalPageRank')
        self.window = window
        self.lda_model = lda_model
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        # define the grammar for selecting the keyphrase candidates
        self.grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        self.kw_extractor = pke.unsupervised.TopicalPageRank()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            grammar=self.grammar,
        )
        num_keywords = len(self.kw_extractor.candidates)


        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        if num_keywords > 0:
            # try:
            self.kw_extractor.candidate_weighting(
                window=self.window,
                pos=self.pos,
                lda_model=self.lda_model,
            )
            # except:
            #     num_keywords = 0

        # get the 10-highest scored candidates as keyphrases
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        elif num_keywords > 0:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)
        else:
            keywords = []

        return keywords


class KeywordsExtractorPositionRank(KeywordsExtractorBase):
    def __init__(self, window=10, max_ngram_size=3, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'PositionRank')
        self.window = window
        self.max_ngram_size = max_ngram_size
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        # define the grammar for selecting the keyphrase candidates
        self.grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        self.kw_extractor = pke.unsupervised.PositionRank()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            grammar=self.grammar,
            maximum_word_number=self.max_ngram_size,
        )
        num_keywords = len(self.kw_extractor.candidates)


        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        if num_keywords > 0:
            # try:
            self.kw_extractor.candidate_weighting(
                window=self.window,
                pos=self.pos,
            )
            # except:
            #     num_keywords = 0

        # get the 10-highest scored candidates as keyphrases
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        elif num_keywords > 0:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)
        else:
            keywords = []

        return keywords


class KeywordsExtractorMultipartiteRank(KeywordsExtractorBase):
    def __init__(self, alpha=1.1, threshold=0.74, method='average', **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get('name', 'MultipartiteRank')
        self.alpha = alpha
        self.threshold = threshold
        self.method = method
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        # define the grammar for selecting the keyphrase candidates
        self.grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"
        self.kw_extractor = pke.unsupervised.MultipartiteRank()

    def _process(self, text):
        # load the content of the document.
        self.kw_extractor.load_document(
            input=text,
            language='en',
        )

        # select the longest sequences of nouns and adjectives, that do
        #    not contain punctuation marks or stopwords as candidates.
        self.kw_extractor.candidate_selection(
            pos=self.pos,
            stoplist=self.stoplist,
        )
        num_keywords = len(self.kw_extractor.candidates)

        # build topics by grouping candidates with HAC (average linkage,
        #    threshold of 1/4 of shared stems). Weight the topics using random
        #    walk, and select the first occuring candidate from each topic.
        if num_keywords > 0:
            try:
                self.kw_extractor.candidate_weighting(
                    alpha=self.alpha,
                    threshold=self.threshold,
                    method=self.method,
                )
            except:
                num_keywords = 0

        # get the 10-highest scored candidates as keyphrases
        if num_keywords > 10:
            keywords = self.kw_extractor.get_n_best(n=10)
        elif num_keywords > 0:
            keywords = self.kw_extractor.get_n_best(n=num_keywords)
        else:
            keywords = []

        return keywords


class KeywordsExtractorRaKUn(KeywordsExtractorBase):
    def __init__(self,
                 distance_threshold=2,
                 distance_method='editdistance',
                 pretrained_embedding_path=None,
                 num_keywords=10,
                 pair_diff_length=2,
                 bigram_count_threshold=2,
                 num_tokens=[1, 2],
                 max_similar=3,
                 max_occurrence=3,
                 **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs.get('name', 'RaKUn')
        self.pos = {'NOUN', 'PROPN', 'ADJ'}
        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += stopwords.words('english')
        # define the grammar for selecting the keyphrase candidates
        self.grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        self.hyperparameters = {
            'distance_threshold': distance_threshold,
            'distance_method': distance_method,
            'pretrained_embedding_path': pretrained_embedding_path,
            'num_keywords': num_keywords,
            'pair_diff_length': pair_diff_length,
            'stopwords': self.stoplist,
            'bigram_count_threshold': bigram_count_threshold,
            'num_tokens': num_tokens,
            'max_similar': max_similar,
            'max_occurrence': max_occurrence,
        }
        self.kw_extractor = mrakun.RakunDetector(self.hyperparameters, verbose=False)

    def _process(self, text):
        # load the content of the document.
        keywords = self.kw_extractor.find_keywords(text, input_type = "text")

        return keywords


class KeywordsExtractorNN(KeywordsExtractorBase):

    class CustomDict(dict):
        def __getattr__(self, item):
            return self[item]

    def __init__(self, config=None, **kwargs):
        super(KeywordsExtractorNN, self).__init__(**kwargs)

        self.name = kwargs.get("name", "Copy-RNN")

        if config is not None:
            self.config = self.CustomDict(config)
        else:
            DEFAULT_CONFIG = json.load(
                open("./NNSupport/config/translators.json", "r", encoding="utf-8")
            )
            self.config = self.CustomDict(DEFAULT_CONFIG)

        self.translator = self._load_model()
        self._ngram = NGram(2)

    def _load_model(self):
        if not os.path.isfile(self.config.models[0]):
            raise FileNotFoundError("No model found, please use the download script in "
                                    "NNSupport/models to download the model.")
        translator = build_translator(self.config)
        return translator

    def _process(self, text: str, title: Union[None, str] = None) -> List[str]:
        """
        :param text: input text
        :param title: After the training, the model has learned to attach great importance to
                              the title, so even if the title is absent, there should also a placeholder
                              for the title.
        :return:
        """
        text = self._preprocess(title, text)
        # print(text)
        input_dict = {
            "id": "7",  # casually selected
            "src": text
        }
        scores, keywords = self.translator.translate(
            [json.dumps(input_dict).encode("utf-8")],
            tgt=['{"id": "7", "tgt": [""]}'.encode("utf-8")],  # tgt must be given or there will be an error
            batch_size=1
        )
        keywords = self._postprocess(keywords[0], text)
        # TODO: need to assign scores to corresponding keywords
        # currently, use a placehold for score
        keywords_scores = [(x, 0.0) for x in keywords]
        return keywords_scores

    def _preprocess(self, raw_title, raw_text):
        """
        tokenize the input text and do some necessary process
        """
        if raw_title is None:
            raw_title = ""
        raw_title = raw_title.strip()
        # raw_title += (raw_title[-1] not in (".", "?", "!")) * "."
        if self.config.lower:
            raw_title = raw_title.lower()
            raw_text = raw_text.lower()
        title_tokens = meng17_tokenize(raw_title)
        text_tokens = meng17_tokenize(raw_text)
        tokens = title_tokens + ["."] + text_tokens
        if self.config.replace_digit:
            tokens = replace_numbers_to_DIGIT(tokens, k=2)
        return " ".join(tokens)

    def _postprocess(self, keywords, text):
        """
        replace unwanted and repeated tokens

        sometimes do guess for <digit>
        """
        text = text.lower()
        keywords = [keyword for keyword in keywords if len(keyword) < 30]
        new_keywords = []
        for keyword in keywords:
            keyword = " ".join([word for word in keyword.split(" ") if re.search(r"\.", word) is None])

            if len(re.sub(r"<unk>|<digit>|\s", "", keyword).strip()) <= 3:
                continue
            elif len(keyword.split(" ")) > 5:
                continue
            if len(re.findall(r"<digit>", keyword)) == 1:
                make_re = keyword.replace("<digit>", r"\d+")
                all_candidate = list(set(re.findall(make_re, text)))
                if len(all_candidate) == 1:
                    keyword = all_candidate[0]
            if re.search(r"<unk>|<digit>", keyword):
                continue
            new_keywords.append(keyword)
        new_new_keywords = []
        for i in range(len(new_keywords)):
            flag = True
            for j in range(len(new_keywords)):
                if i != j and new_keywords[i] in new_keywords[j]:
                    flag = False
                    break
            if flag:
                new_new_keywords.append(new_keywords[i])
        new_keywords = new_new_keywords
        new_new_keywords = []
        for i, keyword in enumerate(new_keywords):
            if i != 0:
                distance = self._ngram.distance(
                    (min(new_keywords[:i], key=lambda x: self._ngram.distance(keyword, x))), keyword
                )
                if distance > 0.1:
                    new_new_keywords.append(keyword)
            else:
                new_new_keywords.append(keyword)

        return new_new_keywords


class KeywordsExtractorCombined(KeywordsExtractorBase):
    def __init__(self,
                 models=[],
                 **kwargs):
        super().__init__(**kwargs)

        self.name = kwargs.get('name', 'Combined')
        self.models = models

        assert self.score_threshold is None

    def _process(self, text):
        keywords = []
        for m in self.models:
            words_and_scores = m._process(text)
            if m.score_threshold:
                words_and_scores = list(filter(
                    lambda x: x[1] > m.score_threshold,
                    words_and_scores
                ))
            keywords.extend(words_and_scores)

        return keywords


def phrase_match(ref_phrase, pred_phrase):
    ref_words = set(ref_phrase.split())
    pred_words = set(pred_phrase.split())
    matched_words = ref_words & pred_words
    ratio_match_ref = len(matched_words) / max(len(ref_words), 1)
    ratio_match_pred = len(matched_words) / max(len(pred_words), 1)
    return ratio_match_ref, ratio_match_pred


def evaluation_a_doc(ref_words, pred_words, ignore_case=True):
    if ignore_case:
        ref_words = [w.lower() for w in ref_words]
        pred_words = [w.lower() for w in pred_words]

    ref_words = set(ref_words)
    pred_words = set(pred_words)

    num_ref = len(ref_words)
    num_pred = len(pred_words)
    matched_values_ref = {w: 0.0 for w in ref_words}
    matched_values_pred = {w: 0.0 for w in pred_words}

    for r_w in ref_words:
        for p_w in pred_words:
            v_ref, v_pred = phrase_match(r_w, p_w)
            if v_ref > matched_values_ref[r_w]:
                matched_values_ref[r_w] = v_ref
            if v_pred > matched_values_pred[p_w]:
                matched_values_pred[p_w] = v_pred

    precision = sum(matched_values_pred.values()) / max(num_pred, 1.0)
    recall = sum(matched_values_ref.values()) / max(num_ref, 1.0)
    F1 = 2*precision*recall/max((precision+recall), 1E-6)
    return precision, recall, F1


def evaluation_many_docs(ref_docs, pred_docs, ignore_case=True):
    assert len(ref_docs) == len(pred_docs)
    all_precision = []
    all_recall = []
    all_f1 = []
    for (ref_phrases, pred_phrases) in zip(ref_docs, pred_docs):
        if len(ref_phrases) == 0:
            continue
        a_precision, a_recall, a_f1 = evaluation_a_doc(
            ref_words=ref_phrases,
            pred_words=pred_phrases,
            ignore_case=ignore_case
        )
        all_precision.append(a_precision)
        all_recall.append(a_recall)
        all_f1.append(a_f1)

    return np.mean(all_precision), np.mean(all_recall), np.mean(all_f1)


def train_word_frequency():
    # stoplist for filtering n-grams
    stoplist = list(string.punctuation)
    # stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    # stoplist += stopwords.words('english')

    # compute df counts and store as n-stem -> weight values
    pke.compute_document_frequency(
        input_dir='../scratch/lda_text',
        output_file='../scratch/tf_abs_2.tsv.gz',
        extension='txt',  # input file extension
        language='en',  # language of files
        normalization="stemming",  # use porter stemmer
        stoplist=stoplist
    )

def keyword_tester(keyword_extractors,
                   in_path='../scratch/paper_samples.json',
                   out_path='../scratch/keywords.html'):
    html_head = ''
    html_body = ''

    all_keywords = {}

    with open(in_path, 'r') as fr:
        data = json.load(fr)

    for doc in data:
        doi = doc['doi']
        abstract = doc['abstract']
        human_keywords = doc['keywords']
        human_keywords = [w.strip() for w in human_keywords]
        human_keywords = list(filter(lambda x: len(x) > 0,  human_keywords))
        if 'human_keywords_full' not in all_keywords:
            all_keywords['human_keywords_full'] = []
        all_keywords['human_keywords_full'].append(human_keywords)

        abstract = KeywordsExtractorBase().clean_html_tag(abstract)
        # if 'body_text' in doc and isinstance(doc['body_text'], list):
        #     for para in doc['body_text']:
        #         abstract+='\n{}'.format(para['Text'])

        if 'human_keywords_in_abs' not in all_keywords:
            all_keywords['human_keywords_in_abs'] = []
        all_keywords['human_keywords_in_abs'].append(list(filter(
            lambda x: x in abstract, human_keywords
        )))

        html_body += '<div style="background-color:#dbe9ea3d; font-size:20px;">\n'
        html_body += '<p>doi: {}</p>\n'.format(doi)
        html_body += '<p>human_keywords: {}</p>\n'.format(', '.join(human_keywords))
        html_body += '<p>{}</p>\n'.format(
            KeywordsExtractorBase().hightlight_keywords(
                keywords=human_keywords,
                text=abstract,
                light_color='#ffea593d',
                deep_color='#ffc107',
            )
        )

        for extractor in keyword_extractors:
            keywords = extractor.process(abstract)
            if extractor.name not in all_keywords:
                all_keywords[extractor.name] = []
            all_keywords[extractor.name].append(keywords)

            html_body += '<p>keyword extractor: {}</p>\n'.format(extractor.name)
            html_body += '<p>keywords: {}</p>\n'.format(', '.join(keywords))
            html_body += '<p>{}</p>\n'.format(
                    extractor.hightlight_keywords(
                    keywords=keywords,
                    text=abstract,
                    light_color='#ffea593d',
                    deep_color='#ffc107',
                )
            )
        html_body += '</div>\n'

    # evaluation
    # compare with human keywords
    for ref_type in ['human_keywords_full', 'human_keywords_in_abs']:
        html_body += '<div style="background-color:#dbe9ea3d; font-size:20px;">\n'
        html_body += '<h2>Compare with {}</h2>\n'.format(ref_type)
        html_body += '''<table>
            <tr>
            <th>Extractor</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            </tr>
        '''
        pred_types = [extractor.name for extractor in keyword_extractors]
        for p_type in pred_types:
            precision, recall, f1 = evaluation_many_docs(
                ref_docs=all_keywords[ref_type],
                pred_docs=all_keywords[p_type],
                ignore_case=True,
            )
            html_body += '''<tr>
                <td>{extractor}</td>
                <td>{precision}</td>
                <td>{recall}</td>
                <td>{f1}</td>
                </tr>
            '''.format(
                extractor=p_type,
                precision=precision,
                recall=recall,
                f1=f1,
            )

        html_body += "</table>\n"
        html_body += '</div>\n'


    with open(out_path, 'w') as fw:
        fw.writelines(
            html_body
        )
    return html_body

def extract_keywords(in_path='../rsc/samples_21181.json',
                     out_path='../scratch/papers_w_keywords.json'):
    with open(in_path, 'r') as fr:
        papers = json.load(fr)

    print('len(papers)', len(papers))

    papers_w_keywords = []
    extractor = KeywordsExtractorRaKUn(
        name='RaKUn_0',
        distance_threshold=2,
        pair_diff_length=2,
        bigram_count_threhold=2,
        num_tokens=[1, 2, 3],
        max_similar=10,
        max_occurrence=3,
        score_threshold=None,
        use_longest_phrase=True,
    )
    for p in papers:
        abstract = KeywordsExtractorBase().clean_html_tag(p['abstract'])
        try:
            keywords = extractor.process(abstract)
            p['keywords'] = keywords
            papers_w_keywords.append(p)
        except:
            print('Error')
            print(abstract)

    with open(out_path, 'w') as fw:
        json.dump(papers_w_keywords, fw, indent=2)


def extract_keywords_in_prodify_format(in_path='../rsc/samples_21181.json',
                                      out_path='../scratch/prodigy_abstracts_21181.jsonl'):
    with open(in_path, 'r') as fr:
        papers = json.load(fr)

    print('len(papers)', len(papers))

    papers_w_keywords = []
    extractor = KeywordsExtractorRaKUn(
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
    nlp = spacy.load('en', disable=['parser', 'ner'])
    for p in papers:
        abstract = KeywordsExtractorBase().clean_html_tag(p['abstract'])
        try:
            keywords = extractor.process(abstract)
        except:
            print('Error')
            print(abstract)
            continue

        highlights = KeywordsExtractorBase().keywords_to_long_highlights(
            text=abstract,
            keywords=keywords,
        )
        doc = nlp(abstract)
        prodigy_spans = KeywordsExtractorBase().text_spans_to_token_spans(
            text_spans=highlights,
            tokens=doc,
            default_label='KEYWORD',
        )
        papers_w_keywords.append({
            'text': p['abstract'],
            'keywords': keywords,
            'spans': prodigy_spans,
            'meta': {'source': p['doi']},
        })

    with open(out_path, 'w') as fw:
        for p in papers_w_keywords:
            json.dump(p, fw)
            fw.write('\n')


def plot_word_cloud(in_path='../scratch/papers_w_keywords.json',
                    out_path='../scratch/keywords_word_cloud.png'):
    with open(in_path, 'r') as fr:
        papers = json.load(fr)
    all_keywords = collections.Counter()
    for p in papers:
        for w in p['keywords']:
            all_keywords[w] += 1

    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stopwords.words('english')
    wordcloud = WordCloud(
        background_color='white',
        max_words=200,
        stopwords=set(stoplist),
        max_font_size=250,
        random_state=30,
        height=860,
        margin=2,
        width=1000,
        collocations=False,
        # mask=alice_coloring
    ).generate_from_frequencies(all_keywords)
    # plt.figure(figsize=(16,10))
    # plt.figure(figsize=(16,10))  
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


if __name__ == '__main__':
    # train_word_frequency()

    # extract_keywords(
    #     in_path='../rsc/samples_21181.json',
    #     out_path='../scratch/papers_w_keywords.json'
    # )

    # extract_keywords_in_prodify_format(
    #     in_path='../rsc/samples_21181.json',
    #     out_path='../scratch/prodigy_abstracts_21181.jsonl'
    # )

    # plot_word_cloud(
    #     in_path='../scratch/papers_w_keywords.json',
    #     out_path='../scratch/keywords_word_cloud.png',
    # )
    #
    keyword_tester(
        keyword_extractors=[
            # KeywordsExtractorSumma(
            #     name='Summa_0',
            #     split=True,
            #     scores=True,
            #     use_longest_phrase=True,
            # ),
            # KeywordsExtractorYake(
            #     name='Yake_20',
            #     max_ngram_size=3,
            #     window_size=1,
            #     top=20,
            #     use_longest_phrase=True,
            # ),
            # KeywordsExtractorRaKUn(
            #     name='RaKUn_0',
            #     distance_threshold=2,
            #     pair_diff_length=2,
            #     bigram_count_threhold=2,
            #     num_tokens=[1,2,3],
            #     max_similar=10,
            #     max_occurrence=3,
            #     score_threshold=None,
            #     use_longest_phrase=True,
            # ),
            # KeywordsExtractorRaKUn(
            #     name='RaKUn_1',
            #     distance_threshold=2,
            #     pair_diff_length=2,
            #     bigram_count_threhold=2,
            #     num_tokens=[1, 2, 3],
            #     max_similar=10,
            #     max_occurrence=3,
            #     num_keywords=20,
            #     score_threshold=None,
            #     use_longest_phrase=True,
            # ),
            # KeywordsExtractorRaKUn(
            #     name='RaKUn_2',
            #     distance_threshold=2,
            #     pair_diff_length=2,
            #     bigram_count_threhold=2,
            #     num_tokens=[1, 2, 3],
            #     max_similar=10,
            #     max_occurrence=3,
            #     num_keywords=30,
            #     score_threshold=None,
            #     use_longest_phrase=True,
            # ),
            # KeywordsExtractorRaKUn(
            #     name='RaKUn_3',
            #     distance_threshold=2,
            #     pair_diff_length=2,
            #     bigram_count_threhold=2,
            #     num_tokens=[1, 2, 3],
            #     max_similar=10,
            #     max_occurrence=3,
            #     score_threshold=0.20,
            #     use_longest_phrase=True,
            # ),
            KeywordsExtractorNN(only_extractive=True),
        ],
        in_path='../scratch/paper_samples.json',
        out_path='../scratch/keywords_test2.html',
    )