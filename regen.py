import argparse

import os
import json
import math
import random
from typing import List, Dict, Tuple
import re
import en_core_web_lg
from evaluation import eva_total
from StructedOutput import ReGeneration, DiffusionSpecifications, Actions
from utils import llm_call

# cut
def cut_specs(specs: List, cut_ratio=0.3, cut_mtd='random'):
    nlp = en_core_web_lg.load()
    if cut_mtd not in ['random', 'md']:
        return "not valid cut method"
    n = len(specs)
    cut_len = math.ceil(cut_ratio * n)
    truncated = [[] for _ in range(n - cut_len + 1)]
    res = [[] for _ in range(n - cut_len + 1)]

    for i in range(n - cut_len + 1):
        cp = specs.copy()
        for j in range(i, i + cut_len):
            trunc_spec = ""
            s = nlp(cp[j])
            if cut_mtd == 'random':  # random 0.2len~0.5len
                rand_int = random.randint(math.floor(0.2 * len(s)), math.floor(0.5 * len(s)))
                trunc_spec = s[:rand_int + 1].text
            if cut_mtd == 'md':  # analysis
                for token in s:
                    if token.tag_ == 'MD':
                        trunc_spec = s[:token.i + 1].text
                        break
            cp[j] = trunc_spec
            truncated[i].append(trunc_spec)
        res[i] = cp
    return res, truncated  # n-len+1 * n, n-len+1 * cut_len


# diffusion
def diffusion(specs, truncated_specs, prompt_info, cut_ratio):
    batch_len = len(specs[0])
    cut_len = math.ceil(cut_ratio * batch_len)
    topic, fun_name, fun_desc = prompt_info
    prompt_files = ("diffusion_sys_msg", "diffusion_usr_msg")
    batch_info = [
        {"n": batch_len,
         "cut_len": cut_len,
         "topic": topic,
         "function_name": fun_name,
         "function_description": fun_desc,
         "specs": "\n".join(specs[i]),
         "truncated": "\n".join(truncated_specs[i])}
        for i in range(len(specs))]

    response = llm_call("diffusion", prompt_files=prompt_files, batch_info=batch_info, schema=DiffusionSpecifications)
    diffusion_result = [[item.completed_truncated_specifications for item in res['parsed'].completedSpecifications] for
                        res in response]
    return diffusion_result


def extract_actions_by_llm(specs):
    prompt_files = ("extract_sys_msg", "extract_usr_msg")
    batch_info = [{"specifications": "\n".join(s)} for s in specs]

    response = llm_call("extract operations", prompt_files=prompt_files, schema=Actions, batch_info=batch_info)
    extracted_actions = [item.__dict__['action'] for res in response for item in res['parsed'].actions]
    extracted_actions = [str(i + 1) + '.' + item for i, item in enumerate(extracted_actions)]
    return extracted_actions


def filter_by_llm(extracted_actions, fun_specs):
    prompt_files = ("extract_sys_msg", "filter_usr_msg")
    batch_info = {"actions": "\n".join(extracted_actions),
                  "specifications": "\n".join(fun_specs)}
    response = llm_call("filter operations", prompt_files=prompt_files, batch_info=batch_info, schema=Actions, batch=False)
    filtered_actions = [item.__dict__['action'] for item in response['parsed'].actions]
    filtered_actions = [str(i + 1) + '.' + item for i, item in enumerate(filtered_actions)]
    return filtered_actions


def save_record(doc, fun, record_info, rp='records'):
    file_path = rp + '/' + doc + "/" + fun + ".json"
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if not os.path.exists(file_path):
        f0 = open(file_path, "w", encoding="utf-8")
        f0.close()
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        file_data = [record_info]
    else:
        with open(file_path, "r", encoding="utf-8") as fr:
            file_data = json.load(fr)
            file_data.append(record_info)
    with open(file_path, "w", encoding="utf-8") as fw:
        json.dump(file_data, fw, ensure_ascii=False, indent=4)


def regen_main(args):
    cut_ratio = args.cut_ratio
    cut_mtd = args.cut_mtd
    is_diff = args.is_diff
    generation = args.generation
    rp = args.rp

    with open("data/re_data.json", "r", encoding="utf-8") as f:
        input_info = json.load(f)
    for doc, cases in input_info.items():
        print("-------------- " + doc + " --------------")
        for fun_spec in cases["specifications"]:
            fun_name = fun_spec["function_name"]
            if re.match(r'.*\d$', fun_name):
                fun_name = fun_name[:-1]
            prompt_info = (cases["topic"], fun_name, fun_spec["function_description"])
            prompt_files = ("regen_sys_msg", "regen_usr_msg")
            if is_diff:
                specs, truncated_specs = cut_specs(fun_spec["function_specifications"], cut_ratio=cut_ratio, cut_mtd=cut_mtd)
                diffusion_result = diffusion(specs, truncated_specs, prompt_info, cut_ratio=cut_ratio)
                action_sequence0 = extract_actions_by_llm(diffusion_result)
                action_sequence = filter_by_llm(action_sequence0, fun_spec["function_specifications"])
            else:
                action_sequence = []
            batch_info = [{"topic": cases["topic"],
                           "function_name": fun_name,
                           "function_description": fun_spec["function_description"],
                           "function_specifications": "\n".join(fun_spec["function_specifications"]),
                           "operation_sequence": "\n".join(action_sequence)
                           } for _ in range(generation)]
            response = llm_call("regen", prompt_files=prompt_files, batch_info=batch_info, schema=ReGeneration)
            print("------" + fun_spec["function_name"] + "------")
            record_info = {
                "desc": "model: {}, generation: {}, isDiffusion: {}".format('gpt-4o', generation, is_diff),
                "diff_act": action_sequence,
                "act_rel": "",  # act rel
                "analysis": {},
                "regen": [],  # n - cut_len + 1
            }
            record_info = record_process(record_info, response)
            save_record(doc, fun_spec["function_name"], record_info, rp=rp)


def record_process(record_info, response):
    # print("regen **** first stage **** results:...")
    for i, res in enumerate(response):
        ana = []
        for j, step in enumerate(res['parsed'].steps):
            ana.append("#step" + str(j + 1) + ": " + step.__dict__['analysis'])
            # print("step" + str(j + 1) + ": ", step.__dict__['analysis'])
        record_info['analysis']["generation" + str(i + 1)] = ana
        final_answer = res['parsed'].final_answer.__dict__
        ans = {
            "generation": str(i + 1),
            "absent_element": final_answer["absent_element"],
            "new_specification": final_answer["new_specification"]
        }
        record_info["regen"].append(ans)
        print("Final answer{}:".format(str(i + 1)))
        print("absent_element: ", final_answer['absent_element'])
        print("new_specification: ", final_answer['new_specification'])
    return record_info


def main():
    parser = argparse.ArgumentParser()
    # method
    parser.add_argument("--cut_ratio", type=float, default=0.3)
    parser.add_argument("--cut_mtd", type=str, default='random')
    parser.add_argument("--is_diff", type=bool, default=True)

    # output
    parser.add_argument("--generation", type=int, default=3)
    parser.add_argument("--rp", type=str, default='records')

    args = parser.parse_args()

    runs = 1
    for _ in range(runs):
        regen_main(args=args)
        eva_total(rp=args.rp)


if __name__ == '__main__':
    main()
