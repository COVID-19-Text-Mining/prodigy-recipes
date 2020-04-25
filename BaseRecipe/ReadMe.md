The base recipe serves for two purpose: 
* Provides most common annotation tasks for COVID study
* You can build more sophisticated recipe based on this and
save some time building wheels.

Gernal usage:
    
    1. Download all the files in the BaseRecipe folder
    2. Go to annotate.covidscholar.org and click to create a new Prodigy instance
    3. Fill Service Name with any name you like
    4. Fill collection name as the MongoDB collection where you want to store the annotation
    5. Fill prodigy arguments as needed. 
    upload all the files to 

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