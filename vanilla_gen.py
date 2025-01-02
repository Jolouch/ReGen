import json
import re

from StructedOutput import VanillaGeneration
from regen import save_record
from utils import llm_call

def gen_vanilla(model='gpt-4o', generation=1, usr_msg='regen_usr_msg_vanilla', rp='records_vanilla', desc='vanilla_prompt'):
    with open("data/re_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for doc, cases in data.items():
        print("-------------- " + doc + " --------------")
        input_info = cases
        prompt_files = ("regen_sys_msg", usr_msg)
        batch_info = [
            {"topic": input_info["topic"],
             "function_name": fun_spec["function_name"][:-1] if re.match(r'.*\d$', fun_spec["function_name"]) else fun_spec["function_name"],
             "function_description": fun_spec["function_description"],
             "function_specifications": "\n".join(fun_spec["function_specifications"]),
             } for fun_spec in input_info["specifications"]]
        response_multiple = []
        for g in range(generation):
            response = llm_call("regen_vanilla", prompt_files=prompt_files, batch_info=batch_info, model=model, temperature=1, schema=VanillaGeneration)
            assert len(input_info["specifications"]) == len(response)
            response_multiple.append(response)
        for i, fun_spec in enumerate(input_info["specifications"]):
            print("------" + fun_spec["function_name"] + "-------")
            record_info = {
                "desc": desc,
                "regen": [],
            }
            assert generation == len(response_multiple)
            for j in range(generation):
                answer = response_multiple[j][i]['parsed'].answers[0]
                ans = answer.__dict__
                tmp = {
                    "generation": str(j + 1),
                    "incompleteness": ans["incompleteness"],
                    "new_specification": ans["new_specification"]}
                record_info["regen"].append(tmp)
                print("---generation " + str(j + 1) + "----")
            save_record(doc, fun_spec["function_name"], record_info, rp=rp)


if __name__ == '__main__':
    gen_vanilla(model="gpt-4o", generation=1, usr_msg='regen_usr_msg_vanilla_cot', rp='records_vanilla_cot@1', desc='vanilla_cot')  # vanilla cot
