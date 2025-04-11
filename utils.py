import time
from typing import Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import configparser

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
