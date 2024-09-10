import os, json
import re
import random
import requests
from time import sleep

"""
    This utils file is strictly for pythonic functions with native libraries,
    embedding_utils file is where utils for third-party library exist
"""


def write_json(data, filepath, mode = "w"):
    with open(filepath, mode) as f:
        json.dump(data, f)


def read_json(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data


def write_jsonl(data, filepath, mode = "w"):

    assert filepath.endswith("jsonl"), f"use .jsonl as extension in your filepath for this function"

    # Write data to the JSONL file
    with open(filepath, mode) as f:
        for item in data:
            json.dump(item, f)  # Write JSON object
            f.write('\n')  # Add newline character after each JSON object


def write_txt(data, path):
    with open(path, 'w') as file:
        # Write each element of the list followed by a newline character
        for item in data:
            file.write(f"{item}\n")


def write_txt_dump(data, path):
    with open(path, 'w') as file:
        file.write(data)


def read_txt(path):
    """
    reads a txt file, assumes last line as source and the rest as content
    """

    with open(path, "r") as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip()
            
    return "".join(data[:-1]).split("."), data[-1]


def read_txt_v2(path):
    """
    reads a txt file, returns the content as a single sentence
    """

    with open(path, "r") as f:
        data = f.readlines()
        for i in range(len(data)):
            data[i] = data[i].strip()
            
    return ".".join(data)



def split_jsonl(input_file, test_set_split = 0.2):

    assert input_file.endswith(".jsonl"), f"expected file with .jsonl extension and not {input_file}"

    with open(input_file, 'r') as infile:
        data = infile.readlines()

    total_lines = len(data)
    sample_size = int(test_set_split * total_lines)

    # Randomly select 20% of the lines
    random_sample = random.sample(data, sample_size)

    _save_path = input_file[:-6]
    print(input_file)
    print(_save_path)
    # Save the 20% to one file
    with open(f"{_save_path}_{str(test_set_split)}.jsonl", 'w') as test_outfile:
        test_outfile.writelines(random_sample)

    # Save the remaining 80% to another file
    remaining_80 = [line for line in data if line not in random_sample]
    with open(f"{_save_path}_{str(1 - test_set_split)}.jsonl", 'w') as train_outfile:
        train_outfile.writelines(remaining_80)


def remove_unicode_escape_sequences(input_string):
    pattern = r'(\\u0[0-9a-fA-F]{1,2})|(\\x[0-9a-fA-F]{1,2})|\xa0'
    cleaned_string = re.sub(pattern, '', input_string)
    return cleaned_string


def parse_steps_stream(text):
    pattern = r'\d+\.\s'
    sections = re.split(pattern, text)
    
    # Remove any empty strings from the resulting list (caused by the split at the start of the text)
    sections = [section for section in sections if section.strip()]
    return sections



def _request_indian_kanoon(params, url = "https://api.indiankanoon.org/search/"):
    signature = os.getenv("INDIANKANOON_API_KEY")

    headers = {
                "Authorization": signature,
                "Content-Type": "application/JSON"
            }
    
    return requests.post(url = url,
                        params = params,
                        headers = headers).json()


def find_caselaw(pagenum = 0, blacklist = ["kwords", "doctypes"], **kwargs):
    
    # kwords = kwargs["kwords"]
    kwords = kwargs["kwords"].strip()
    
    if isinstance(kwords, str):
        if "," in kwords:
            kwords = kwords.split(",")
        else:
            kwords = [kwords]

    terms = " ORR ".join(kwords)

    doctypes = kwargs.get("doctypes", "judgments")
    if isinstance(doctypes, list):
        doctypes = ",".join(doctypes)

    terms += " doctypes:" + doctypes # make sure to keep the space before and after ORR.

    for _arg in kwargs.items():
        _key, _value = _arg
        if _key not in blacklist:
            terms += f" {_key}:{_value}" # the leading space is required

    print(terms)
    print()
    
    payload = {
        "formInput": terms,
        "pagenum": pagenum,
        **kwargs
    }

    response = _request_indian_kanoon(params = payload)
    
    retrieved_context, citations = _process_caselaws(response)
    return retrieved_context, citations


def search_doc_id(doc_id):
    # doc_id = "1851149"
    # url = f"https://api.indiankanoon.org/doc/{doc_id}/?maxcites={10}&maxcitedby={20}"
    
    url = f"https://api.indiankanoon.org/doc/{doc_id}/"
    
    return _request_indian_kanoon(params = {}, url = url)


def explain_caselaw(case_title, use_scrapper = False):

    # setting use_scrapper to False is using indian kanoon api to retrieve document content
    print(f"trying to find title: {case_title}")
    payload = {
        "formInput": f"title:{case_title}"
    }
    response = _request_indian_kanoon(params = payload)
    retrieved_context, citations = _process_caselaws(response) # retrieved_context is kind of redundant
    print(f"got context: {retrieved_context} \n")
    if citations:
        if use_scrapper:
            # use requests.get() and appropriately scrape it
            pass
        else:
            doc_id = str(citations[0].split("/")[-1])
            response = search_doc_id(doc_id = doc_id)
            print(f"got document {response}")
            retrieved_context = remove_html_tags(response["doc"])
            return retrieved_context, [citations[0]]
    
    else:
        # nothing was found with that title
        return "**There were no results found for the given query**", []


def _process_caselaws(response):
    
    retrieved_context = ""

    # response to be of indian kanoon's response format
    docs = response["docs"]
    citations, case_titles, headlines = [], [], []
    for i, doc in enumerate(docs):

        doc_id = str(doc["tid"])
        
        if doc_id not in citations:
            # meant for getting only unique cases
            citations.append(f"https://indiankanoon.org/doc/{doc_id}")
            case_titles.append(doc["title"])
            headlines.append(doc["headline"])

    
    for i, (case_title, headline, citation) in enumerate(zip(case_titles, headlines, citations)):
        print(f"USING CITATION: {citation}")
        retrieved_context += f"{i+1}. {case_title}: {headline}\n source:{citation} \n\n" # keep the \n because it would add the next 'i+1' right after and give you the wrong link

    print(f"RETRIEVED THE FOLLOWING FROM INDIAN KANOON WITH SOURCES (MAKE SURE TO CITE THEM IN MARKDOWN FORMAT): {retrieved_context}")

    if retrieved_context:
        return retrieved_context, citations
    
    else:
        # retrieved_context, citations
        return "**There were no results found for the given query**", []


def google_search(query):
    key = os.getenv("GOOGLE_SEARCH_API_KEY")
    _cx = os.getenv("GOOGLE_CX_KEY")

    print(key, _cx)

    url = "https://www.googleapis.com/customsearch/v1"
    payload = {
        "key": key,
        "q": query,
        "cx": _cx,
        "cr": "countryIN"
    }
    response = requests.get(url, 
                            params = payload
                            )
    print(response)
    return response


def extract_links(response_message):
    # Regular expression to extract URLs from text    
    pattern = r'https?://[^\s>]+'
    matches = re.findall(pattern, response_message)
    print("Extracted URLs:", matches)
    return matches


def remove_html_tags(text):
    # regex to remove anything in between html tags. ie between <xyz> and </xyz>
    clean_text = re.sub(r'</?[^>]+>', '', text)
    return clean_text


def analyse_stream(word_stream, stream_def, look_for_defs, current_behaviour, found_closing_tag = False):

    for _def in look_for_defs:
        if _def in word_stream:
            found_def = re.search(_def, word_stream)
            if found_def and not found_closing_tag:
                x, y = found_def.span()
                closing_tag = stream_def[_def]["closing_tag"]
                behaviour = stream_def[_def]["behaviour"]

                return {
                    "indices": [x, y],
                    "closing_tag": closing_tag,
                    "behaviour": behaviour
                   }

            elif found_def and found_closing_tag:
                # print(f"found {_def}")
                # what happens when you find the closing tag
                x, y = found_def.span()
                return {
                    "indices": [x, y],
                    "closing_tag": "",
                    "behaviour": current_behaviour
                   }

    return {}


def process_analysis(analysis, stream_def, return_behaviour = False):
    x, y = analysis["indices"]
    closing_tag = analysis["closing_tag"]

    found_closing_tag = True if closing_tag else False # can be renamed to more like -> need_to_find_closing_tag
    look_for_defs = [closing_tag] if found_closing_tag else list(stream_def.keys())
    behaviour = analysis["behaviour"]

    return (x, y, closing_tag, found_closing_tag, look_for_defs, behaviour) if return_behaviour else (x, y, closing_tag, found_closing_tag, look_for_defs)


def augment_prompt(init_query, augmentation_prompt, retrieved_context):
    if not augmentation_prompt.endswith(":"):
        augmentation_prompt += ":"
    return f"{init_query} \n\n {augmentation_prompt} \n {retrieved_context}"


def format_search_results(query, steps, search_results, whitelisted_string = ""):

    if whitelisted_string:
        formatted_str = f"The primary query is as follows: {query}. Here is some additional context <context> {whitelisted_string} </context> \n\nGiven below are questions and answers - which come from an online factual source.\n\n"
    else:
        formatted_str = f"The primary query is as follows: {query} \n\nGiven below are questions and answers - which come from an online factual source.\n\n"

    # citations = []

    for i, (step, search_result) in enumerate(zip(steps, search_results)):
        # _result, citation = search_result
        _result = search_result
        
        _links = extract_links(_result)
        # citations.extend(_links)

        formatted_str += f"{i + 1}. {step}: \n<context> {_result} </context>\n\n"
        
        if _links:
            formatted_str += "<references>" + "".join(_links) + "</references>"
            # for _link in _links:
            #     formatted_str += "<reference>" + _link + "</reference>"


    formatted_str += "Answer the primary query based on the retrieved context. Do not forget to mention the all source URLs for the same separated by '\n' if there are many"
    # return formatted_str, list(set(citations))
    return formatted_str



def retry(func):
    max_retries = 5
    def wrapper(*args, **kwargs):
        for i in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except:
                _sleep_time = 5
                sleep(_sleep_time)
                print(f"failed {i}th time mostly due to rate limits; sleep for {_sleep_time}s; q was: {args}")

    return wrapper



if __name__ == "__main__":
    input_file = "./results/paraphrased_refined.jsonl"
    # split_jsonl(input_file, test_set_split=0.15)
    google_search(input_file)

