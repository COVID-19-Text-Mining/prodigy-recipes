##### The base recipe serves for two purpose: 
* Provides most common annotation tasks for COVID study.
* You can build more sophisticated recipe based on this and
save some time building wheels.

##### Gernal usage:
    
* Download all the files in the BaseRecipe folder.
* Go to `annotate.covidscholar.org` and click to create a new Prodigy instance.
* Fill Service Name with any name you like.
* Fill collection name as the MongoDB collection where you want to store the annotation.
* Fill prodigy arguments as needed. 
  For this BaseRecipe particularlly, you can use
  `COVIDBase -F COVIDBase.py -task-type ner,textcat,note -ner-label vaccine,disease 
  -textcat-label experimental,theoretical`. 
  `COVIDBase` is the name of recipe. 
  `-F COVIDBase.py` tells `Prodigy` to load the recipe from `COVIDBase.py`.
  The rest are arguments defined in the recipe decorator. 
  See more introduction in the following part. 
* Upload all the files in the BaseRecipe folder.
* Click `Save` and then `Start`.
* If everything is working, you should see `Start annotation`. 
  Otherwise, there might be an error and you can check it via `View Console`.    
* Share the task to others via `Sharing`.

##### How to specify data source:

There are two ways to load data.

* Load from database (recommended). 
  Specify `dataset_name` as the collection name in MongoDB.
  The default value is `-dateset-name entries`.

* Load from .jsonl file. 
  Many formats are supported. 
  However, .jsonl is the one with the best performance for `Prodigy`.
  Specify `dataset_file` as the name of data file and assign a custom name via `dataset_name`.
  E.g.: `-dataset-name ExampleData -dataset-file examples.jsonl`.

##### How to specify tasks:

Currently, there are four default tasks you can choose among.
You can specify `task-type` to select.
The default value is `-task-type ner, textcat, summary, note`.
* NER (Named-Entity Recognition): mark the words or phrase in text with different labels.
    You can define the labels with `ner-label`. E.g. `-ner-label vaccine,disease`.
* TextCat (Text categorization): choose the categories the paragraph falls in.
    Multiple choice is enabled by default.
    You can define the labels with `textcat_label`. E.g. `-textcat-lebel mechanism,diagnostics`.
    The default value is eleven classes defined by our Expert, Kevin.
* Summary: Summarize the paragraph with more consice sentences.
* Note: Add any note you like, such as summary, important points, critical parameters, etc.

##### How to add custom html, javascript and css

If you want to design your own html, javascript, and css
load them from file as following and pass them to corresponding interface.

    with open('keywords_annotation.html') as txt:
        template_text = txt.read()
    with open('keywords_annotation.js') as txt:
        script_text = txt.read()
    with open('keywords_annotation.css') as txt:
        css_text = txt.read()

* Html. You can create a block with `view_id` of `html` and pass the `template_text` to `html_template`.
  Then, add this block to all_task_blocks. 
  E.g.:

        all_task_blocks.append({
            'view_id': 'html',
            'html_template': template_text,
        })
    
    
* Javascript. Pass `script_text` to `javascript` when returning.
  If you are familiar with `React`, you can also modify `bundle.js` as a more flexible way. 
  E.g.:
  
        return {
            ...
            "config": {  # Additional config settings, mostly for app UI
                ...
                'javascript': script_text,  # custom js
                ...
            },
            ...
        }
* CSS. There are two ways.
    * Add the CSS in `index.html`.
        E.g.:
        
            <head>
                ...
                <style>
                    .prodigy-content {
                        text-align: justify !important;
                    }
                </style>
                ...
            </head>
    * Pass `css_text` to `global_css` when returning. 
    However, this is slower than modifying `index.html` directly.
    E.g.:

            return {
                ...
                "config": {  # Additional config settings, mostly for app UI
                    ...
                    'css_text': script_text,  # custom js
                    ...
                },
                ...
            }
