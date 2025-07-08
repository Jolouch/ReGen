import json
import os
import re
import time
from typing import Dict, List, Tuple

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import configparser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

config = configparser.ConfigParser()
config.read('setting.ini')
api_key = config['API']['api_key']

def llm_call(process: str, prompt_files: Tuple, batch_info, model='gpt-4o', temperature=1, batch=True, schema=None, include_raw=True):
    sys_msg_filename, usr_msg_filename = prompt_files
    sys_msg = open("prompt/" + sys_msg_filename, encoding='utf8').read()
    usr_msg = open("prompt/" + usr_msg_filename, encoding='utf8').read()
    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_msg),
        ("user", usr_msg)
    ])
    count = 0
    while True:
        try:
            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
            )
            print(process + " processing, llm calling..., trying: " + str(count + 1))
            if schema is None:
                regen_chain = prompt | llm
            else:
                regen_chain = prompt | llm.with_structured_output(schema=schema, include_raw=include_raw)
            if batch:
                response = regen_chain.batch([item for item in batch_info])
            else:
                response = regen_chain.invoke(batch_info)
            return response
        except Exception as e:
            if count > 5:
                print("cannot get response from remote:", e)
                break
            time.sleep(7)
            print("llm calling failed, retrying...")
            count += 1

def action_simi():
    model = "all-mpnet-base-v2"
    rp = "results/rq1/records_regen@3_1"

    actions = []
    for dirpath, _, files in os.walk(rp):
        for file in files:
            path = os.path.join(dirpath, file)
            with open(path, 'r', encoding='utf-8') as f:
                records_data = json.load(f)
                act = []
                for item in records_data[-1]['diff_act']:
                    tmp = re.sub(r'^\d+\.\s*', '', item)
                    act.append(tmp)
                actions.append(act)

    model = SentenceTransformer(model)
    similarities = []
    for action in actions:
        n = len(action)
        if n > 1:
            word_vectors = model.encode(action, convert_to_tensor=True)
            vectors_np = word_vectors.numpy()
            similarity_matrix = cosine_similarity(vectors_np)
            average_similarity = np.sum(np.triu(similarity_matrix, k=1)) / (n * (n - 1) / 2)
            similarities.append(round(average_similarity, 2))
    print("average semantic similarity of actions: {}".format(np.mean(similarities)))
