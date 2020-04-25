//Using {} around all the code to avoid global variable exposed to outside.
{
// import React, { useState, useContext, useRef, useEffect } from "react";

// global variables
    let keywords_lock = false
    let locked_task_hash = undefined
    let last_spans = []

// functions from external sources
// ------------------ external functions start ----------------------
// it is weird that typeof and instanceof does not work well for dict (object) and array
// therefore, we use Array.isArray for array and trueTypeOf() for others
// ref: https://stackoverflow.com/questions/203739/why-does-instanceof-return-false-for-some-literals
    function trueTypeOf(value) {
        return Object.prototype.toString.call(value).slice(8, -1);
    }

// ref: https://stackoverflow.com/questions/7837456/how-to-compare-arrays-in-javascript
    function equal_object(object_1, object_2) {
        //For the first loop, we only check for types
        for (propName in object_1) {
            //Check for inherited methods and properties - like .equals itself
            //https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/hasOwnProperty
            //Return false if the return value is different
            if (object_1.hasOwnProperty(propName) != object_2.hasOwnProperty(propName)) {
                return false;
            }
            //Check instance type
            else if (typeof object_1[propName] != typeof object_2[propName]) {
                //Different types => not equal
                return false;
            }
        }
        //Now a deeper check using other objects property names
        for (propName in object_2) {
            //We must check instances anyway, there may be a property that only exists in object_2
            //I wonder, if remembering the checked values from the first loop would be faster or not 
            if (object_1.hasOwnProperty(propName) != object_2.hasOwnProperty(propName)) {
                return false;
            } else if (typeof object_1[propName] != typeof object_2[propName]) {
                return false;
            }
            //If the property is inherited, do not check any more (it must be equa if both objects inherit it)
            if (!object_1.hasOwnProperty(propName))
                continue;

            //Now the detail check and recursion

            //This returns the script back to the array comparing
            /**REQUIRES Array.equals**/
            if (Array.isArray(object_1[propName])
                && Array.isArray(object_2[propName])) {
                // recurse into the nested arrays
                if (!equal_array(object_1[propName], object_2[propName]))
                    return false;
            } else if (trueTypeOf(object_1[propName]) == 'Object'
                && trueTypeOf(object_2[propName]) == 'Object') {
                // recurse into another objects
                //console.log("Recursing to compare ", object_1[propName],"with",object_2[propName], " both named \""+propName+"\"");
                if (!equal_object(object_1[propName], object_2[propName]))
                    return false;
            }
            //Normal value comparison for strings and numbers
            else if (object_1[propName] != object_2[propName]) {
                return false;
            }
        }
        //If everything passed, let's say YES
        return true;
    }

    function equal_array(array_1, array_2) {
        // if the other array is a falsy value, return
        if (!array_1 || !array_2)
            return false;

        // compare lengths - can save a lot of time
        if (array_1.length != array_2.length)
            return false;

        for (var i = 0, l = array_1.length; i < l; i++) {
            // Check if we have nested arrays
            if (Array.isArray(array_1[i]) && Array.isArray(array_2[i])) {
                // recurse into the nested arrays
                if (!equal_array(array_1[i], array_2[i]))
                    return false;
            }
            /**REQUIRES OBJECT COMPARE**/
            else if (trueTypeOf(array_1[i]) == 'Object'
                && trueTypeOf(array_2[i]) == 'Object') {
                if (!equal_object(array_1[i], array_2[i]))
                    return false;
            } else if (array_1[i] != array_2[i]) {
                // Warning - two different object instances will never be equal: {x:20} != {x:20}
                return false;
            }
        }
        return true;
    }

// ref: https://stackoverflow.com/questions/3446170/escape-string-for-use-in-javascript-regex
    function escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // $& means the whole matched string
    }

// ------------------ external functions end ----------------------

    function get_keywords_from_spans(spans, text) {
        let all_keywords = []

        let to_add = undefined
        for (s in spans) {
            to_add = true
            span_text = text.substring(spans[s].start, spans[s].end).trim()
            span_text_lower = span_text.toLowerCase()
            for (k in all_keywords) {
                if (span_text_lower == all_keywords[k].toLowerCase()) {
                    to_add = false
                    break
                }
            }
            if (to_add == true) {
                all_keywords.push(span_text)
            }
        }
        return all_keywords
    }

    function get_canceled_keywords_set(old_spans, new_spans, text) {
        let keywords_to_remove = new Set()

        let to_remove = undefined
        for (i in old_spans) {
            to_remove = true
            for (j in new_spans) {
                if (old_spans[i].start >= new_spans[j].start
                    && old_spans[i].end <= new_spans[j].end) {
                    to_remove = false
                    break
                }
            }
            if (to_remove == true) {
                keywords_to_remove.add(
                    text.substring(old_spans[i].start, old_spans[i].end).trim()
                )
            }
        }
        return keywords_to_remove
    }

    function get_spans_from_keywords(keywords, text, label) {
        let all_spans = []

        // get all matched in original text
        let all_highlights = []
        let pattern_key = undefined
        let len_key = undefined
        for (k in keywords) {
            len_key = keywords[k].length
            pattern_key = new RegExp(escapeRegExp(keywords[k]), 'gi')
            while ((match = pattern_key.exec(text)) != null) {
                all_highlights.push({
                    'start': match.index,
                    'end': match.index + len_key,
                    'text': keywords[k],
                    'label': label,
                })
            }
        }
        all_highlights.sort((a, b) => (a.start > b.start) ? 1 : -1)

        let tmp_start = undefined
        let tmp_end = undefined
        // connect adjecent keywords
        for (h in all_highlights) {
            if (all_spans.length == 0) {
                all_spans.push(all_highlights[h])
                continue
            }
            tmp_start = all_spans[all_spans.length - 1].end
            tmp_end = all_highlights[h].start
            if (tmp_end <= tmp_start
                || text.substring(tmp_start, tmp_end).trim() == '') {
                // merge
                all_spans[all_spans.length - 1].end = Math.max(
                    all_spans[all_spans.length - 1].end,
                    all_highlights[h].end
                )
                all_spans[all_spans.length - 1].text = text.substring(
                    all_spans[all_spans.length - 1].start,
                    all_spans[all_spans.length - 1].end
                )
            } else {
                all_spans.push(all_highlights[h])
            }
        }
        return all_spans
    }

    function text_spans_to_token_spans(text_spans, tokens) {
        let token_spans = []

        let text_tokens = undefined
        for (i in text_spans) {
            text_tokens = tokens.filter(
                token => {
                    return (token.start >= text_spans[i].start
                        && token.end <= text_spans[i].end)
                }
            )
            if (text_tokens.length > 0
                && text_tokens[0].start == text_spans[i].start
                && text_tokens[text_tokens.length - 1].end == text_spans[i].end
            ) {
                token_spans.push({
                    'start': text_spans[i].start,
                    'end': text_spans[i].end,
                    'label': text_spans[i].label,
                    'token_start': text_tokens[0].id,
                    'token_end': text_tokens[text_tokens.length - 1].id,
                })
            }
        }
        return token_spans
    }

    function activate_lock() {
        keywords_lock = true
        locked_task_hash = window.prodigy.content._task_hash
    }

    function deactivate_lock() {
        keywords_lock = false
        locked_task_hash = undefined
    }

    function update_keywords(task_detail = undefined) {
        if (keywords_lock == true
            && locked_task_hash == window.prodigy.content._task_hash) {
            return
        } else {
            activate_lock()
        }
        const label = 'KEYWORD'
        const text = window.prodigy.content.text
        const tokens = window.prodigy.content.tokens
        let current_spans = undefined
        if (task_detail) {
            current_spans = task_detail.spans
        } else {
            current_spans = window.prodigy.content.spans
        }
        if (current_spans == undefined) {
            current_spans = []
        }
        current_spans.sort((a, b) => (a.start > b.start) ? 1 : -1)
        // console.log('current_spans', current_spans)
        if (equal_array(current_spans, last_spans)) {
            setTimeout(deactivate_lock, 50)
            // deactivate_lock()
            return
        }
        // get unified keywords
        let current_keywords = get_keywords_from_spans(current_spans, text)
        let keywords_to_remove = get_canceled_keywords_set(last_spans, current_spans, text)
        let keywords_to_remove_lower = new Set()
        keywords_to_remove.forEach(
            k => keywords_to_remove_lower.add(k.toLowerCase())
        )
        current_keywords = current_keywords.filter(
            k => {
                return (!keywords_to_remove_lower.has(k.toLowerCase()))
            }
        )
        // highlight all keywords even not annotated
        let new_spans = get_spans_from_keywords(current_keywords, text, label)
        new_spans = text_spans_to_token_spans(new_spans, tokens)
        // update keywords by connecting continuous tokens
        let new_keywords = get_keywords_from_spans(new_spans, text)
        // console.log('new_keywords', new_keywords)
        last_spans = new_spans
        window.prodigy.update({
            spans: new_spans,
            keywords: new_keywords,
        })

        const taskHash = window.prodigy.content._task_hash
        window.prodigy.update({_task_hash: 0})
        window.prodigy.update({_task_hash: taskHash})

        setTimeout(deactivate_lock, 50)
        // deactivate_lock()
    }

    document.addEventListener('prodigyupdate', event => {
        setTimeout(function () {
            update_keywords(event.detail.task)
        }, 50);
    })

}