import argparse
import json
import os
import time
from collections import Counter
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import configparser

from utils import action_simi

config = configparser.ConfigParser()
config.read('setting.ini')
api_key = config['API']['api_key']

def get_gen(label_data, doc, rp='records'):
    """prepare data for evaluation
    @param label_data: label
    @param doc: requirements document
    @param rp: record path
    @return: formatted data for evaluation
    """
    function_output = []
    for _, _, files in os.walk(rp + "/" + doc):
        function_output = files
    eva_data = {}
    for fun_file in function_output:
        fun = fun_file.split(".")[0]
        eva_data[fun] = {"specs": "", "label": "", "absence": "", "predictions": []}
        for spec in label_data["specifications"]:
            if spec["function_name"] == fun:
                eva_data[fun]["specs"] = spec["function_specifications"]
                eva_data[fun]["label"] = spec["label"]
                eva_data[fun]["absence"] = spec["absence"]
                break

        with open(rp + "/" + doc + "/" + fun_file, 'r', encoding='utf-8') as f:
            regen_data = json.load(f)
        for answers in regen_data[-1]["regen"]:
            if 'vanilla' not in rp:  #
                eva_data[fun]['predictions'].append(answers["absent_element"])  # absent_element, new_specification;
            else:
                eva_data[fun]['predictions'].append(answers["incompleteness"])  # incompleteness, new_specification
    return eva_data


def accuracy(llm_eva_results):
    t = 0
    for item in llm_eva_results:
        if 'true' in item[1]:
            t += 1
    return t, len(llm_eva_results)


def fmt_eva_res(eva_res_str):
    eva_res_str = eva_res_str.lower()
    return 'true' if 'true' in eva_res_str else 'false'


def eva(eva_data, model='gpt-4o', temperature=1):
    """call llm for evaluation

    @param eva_data: data for evaluation
    @param model: llm
    @param temperature: temperature
    @return: evaluation results for 5 times: [('true', 3), ('false', 2)]
    """
    eva_usr_msg = open('prompt/eva_usr_msg', 'r', encoding='utf-8').read()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an excellent AI assistant."),
        ("user", eva_usr_msg)
    ])
    count = 0
    result = []
    for fun, values in eva_data.items():
        while True:
            try:
                print("regen evaluation processing, eva fun: {}, llm calling..., trying: {}".format(fun, count + 1))
                llm = ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                )
                eva_chain = prompt | llm
                final_anss = []
                for ans in values['predictions']:
                    response = eva_chain.batch([
                        {
                            "specs": "\n".join(values["specs"]),
                            "label": values["label"],
                            "absence": values["absence"],
                            "prediction": ans,
                        } for _ in range(5)])
                    counts = Counter([fmt_eva_res(res.content) for res in response])
                    # print(counts)
                    final_anss.append(counts.most_common(n=1)[0][0])  # [('true', 2), ('false', 1)]
                result.append((fun, final_anss))
                break
            except Exception as e:
                if count > 5:
                    print("cannot get response from remote:", e)
                    break
                time.sleep(7)
                print("llm calling failed, retry...")
                count += 1
    return result


def eva_total(rp, model="gpt-4o"):
    """evaluate for all documents

    @param rp: record path
    @param model: llm
    @return: global accuracy rate
    """
    total_true = 0
    total_samp = 0
    with open("data/re_data.json", "r", encoding="utf-8") as f:
        re_data = json.load(f)
    for doc in os.listdir(rp):
        print("------eva doc: {}---------".format(doc))
        eva_data = get_gen(re_data[doc], doc, rp)
        # sm = simi(eva_data)
        llm_sm = eva(eva_data, model=model)
        for i, res in enumerate(llm_sm):
            # print(llm_sm[i], sm[i])
            with open(rp + "/" + doc + "/" + res[0] + ".json", 'r', encoding='utf-8') as fr:
                regen_data = json.load(fr)
            llm_eva_results = [1 if item == 'true' else 0 for item in res[1]]
            regen_data[-1]["label"] = eva_data[res[0]]["label"]
            regen_data[-1]["absence"] = eva_data[res[0]]["absence"]
            # regen_data[-1]['semantic_similarity'] = ",".join(map(str, sm[i][1]))
            regen_data[-1]['llm_eva_results'] = ",".join(map(str, llm_eva_results))
            regen_data[-1]['human_eva_results'] = ""
            with open(rp + "/" + doc + "/" + res[0] + ".json", "w", encoding="utf-8") as fw:
                json.dump(regen_data, fw, ensure_ascii=False, indent=4)
        total_true += accuracy(llm_sm)[0]
        total_samp += accuracy(llm_sm)[1]
    print("\n accuracy: {:.2f}%".format(total_true / total_samp * 100))


def statistics():
    count = {"1": 0, "2": 0, "3": 0}
    with open("data/re_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    for cases in data.values():
        for item in cases["specifications"]:
            if int(item["sample_level"]) == 1:
                count["1"] += 1
            if int(item["sample_level"]) == 2:
                count["2"] += 1
            if int(item["sample_level"]) == 3:
                # print(file, item['function_name'])
                count["3"] += 1
    return count


def acc_sample_levels(rp):
    """evaluate accuracy rate across three levels
    @param rp: record path
    @return: results: {}
    """
    count = statistics()
    acc_count = {"1": 0, "2": 0, "3": 0}
    with open("data/re_data.json", "r", encoding="utf-8") as f:
        input_info = json.load(f)
    for dirpath, _, files in os.walk(rp):
        for file in files:
            path = os.path.join(dirpath, file)
            with open(path, 'r', encoding='utf-8') as f:
                records_data = json.load(f)
            level = 0
            label_data = input_info[path.split("\\")[1]]
            for spec in label_data["specifications"]:
                if spec["function_name"] == file.split(".")[0]:
                    level = spec["sample_level"]
            llm_eva_results = records_data[-1]["llm_eva_results"].split(",")
            if '1' in llm_eva_results:
                acc_count[str(level)] += 1
    acc_total = acc_count['1'] + acc_count['2'] + acc_count['3']
    total_samples = count['1'] + count['2'] + count['3']
    results = {"all": round(acc_total / total_samples * 100, 2), "l1": round(acc_count["1"] / count["1"] * 100, 2),
               "l2": round(acc_count["2"] / count["2"] * 100, 2), "l3": round(acc_count["3"] / count["3"] * 100, 2)}
    return results


def eva_llm_human(rp):
    """assess accuracy rate of llm evaluations

    @param rp: record path
    @return: results:{}
    """
    llm_eva_results = []
    human_eva_results = []
    json_files = []
    for root, _, files in os.walk(rp):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            regen_data = json.load(f)
            llm_eva_results.extend(list(map(int, regen_data[-1]["llm_eva_results"].split(","))))
            human_eva_results.extend(list(map(int, regen_data[-1]["human_eva_results"].split(","))))
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy_score = accuracy_score(human_eva_results, llm_eva_results)
    precision_score = precision_score(human_eva_results, llm_eva_results)
    recall_score = recall_score(human_eva_results, llm_eva_results)
    f1_score = f1_score(human_eva_results, llm_eva_results)
    results = {"accuracy_score": round(accuracy_score * 100, 2), "precision_score": round(precision_score * 100, 2),
               "recall_score": round(recall_score * 100, 2), "f1_score": round(f1_score * 100, 2)}
    return results


def eva_act_rel(rp):
    """evaluate valuable actions

    @param rp: record path
    @return: results:{}
    """
    act_rel = 0
    true_pred = 0
    total_true_case = 0
    json_files = []
    for root, _, files in os.walk(rp):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    action = 0
    useful_actions = 0
    useful_actions_coverage = 0
    total_case = len(json_files)
    for case in json_files:
        with open(case, 'r', encoding='utf-8') as f:
            regen_data = json.load(f)
            act_rel_seg = list(map(int, regen_data[-1]["act_rel"].split(",")))  # [1,0,0]
            assert len(act_rel_seg) == len(regen_data[-1]['diff_act'])
            action += len(regen_data[-1]['diff_act'])
            useful_actions += Counter(act_rel_seg)[1] + Counter(act_rel_seg)[-1]

            llm_eva_results = list(map(int, regen_data[-1]["llm_eva_results"].split(",")))
            if 1 in act_rel_seg:
                useful_actions_coverage += 1
                if 1 in llm_eva_results:
                    true_pred += 1
            if 1 in llm_eva_results:
                total_true_case += 1
                if 1 in act_rel_seg:
                    act_rel += 1
    results = {
        "m1": round(useful_actions / action * 100, 2),
        "m2": round(useful_actions_coverage / total_case * 100, 2),
        "m3": round(true_pred / useful_actions_coverage * 100, 2),
        "m4": round(act_rel / total_true_case * 100, 2),
    }
    return results


def eva_d_m(rp, k=3):
    """evaluate D-M

    @param rp:
    @param k: generation
    @return:
    """
    json_files = []
    useless_gen = 0
    m_c = 0
    d_m = 0
    for root, _, files in os.walk(rp):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    total_gen = len(json_files) * k
    for case in json_files:
        with open(case, 'r', encoding='utf-8') as f:
            regen_data = json.load(f)
        dm = list(map(int, regen_data[-1]["D-M"].split(",")))
        assert len(dm) == k
        cnt = Counter(dm)
        useless_gen += cnt[0]
        m_c += cnt[-1]
        d_m += cnt[1]
    return round(d_m / (total_gen - m_c) * 100, 2)


def comt_avg(path_list, eva='mc', k=3):
    results = {"all": 0, "l1": 0, "l2": 0, "l3": 0}
    if eva == 'mc':
        for path in path_list:
            tmp = acc_sample_levels(path)
            results["all"] += tmp["all"]
            results["l1"] += tmp["l1"]
            results["l2"] += tmp["l2"]
            results["l3"] += tmp["l3"]
        results["all"] = round(results["all"] / 3, 2)
        results["l1"] = round(results["l1"] / 3, 2)
        results["l2"] = round(results["l2"] / 3, 2)
        results["l3"] = round(results["l3"] / 3, 2)
        return results
    else:
        results_dm = 0.
        for path in path_list:
            tmp = eva_d_m(path, k)
            results_dm += tmp
        return round(results_dm / 3, 2)


def main(args):
    if args.rq == "rq1":
        # ReCompGPT pass@1 and pass@3
        results_1_paths = ["results/rq1/records_regen@1_" + str(i+1) for i in range(3)]
        results_3_paths = ["results/rq1/records_regen@3_" + str(i+1) for i in range(3)]
        results_1 = comt_avg(results_1_paths)
        results_3 = comt_avg(results_3_paths)
        # ReCompGPT local pass@1 and pass@3
        results_1_local_paths = ["results/rq1/records_local@1_" + str(i+1) for i in range(3)]
        results_3_local_paths = ["results/rq1/records_local@3_" + str(i+1) for i in range(3)]
        results_1_local = comt_avg(results_1_local_paths)
        results_3_local = comt_avg(results_3_local_paths)
        # human eva
        llm_human = eva_llm_human(results_3_paths[0])
        print("ReCompGPT@1 ALL: {}, L1: {}, L2: {}, L3: {}"
              .format(results_1['all'], results_1['l1'], results_1['l2'], results_1['l3']))
        print("ReCompGPT@3 ALL: {}, L1: {}, L2: {}, L3: {}"
              .format(results_3['all'], results_3['l1'], results_3['l2'], results_3['l3']))
        print(
            "ReCompGPT@1_LOCAL ALL: {}, L1: {}, L2: {}, L3: {}"
            .format(results_1_local['all'], results_1_local['l1'], results_1_local['l2'], results_1_local['l3']))
        print(
            "ReCompGPT@3_LOCAL ALL: {}, L1: {}, L2: {}, L3: {}"
            .format(results_3_local['all'], results_3_local['l1'], results_3_local['l2'], results_3_local['l3']))

        print("human evaluation: ", llm_human)

    if args.rq == "rq2":
        wo_diff_1 = comt_avg(["results/rq2/records_regen_wo_diff@1_" + str(i+1) for i in range(3)])
        wo_diff_3 = comt_avg(["results/rq2/records_regen_wo_diff@3_" + str(i+1) for i in range(3)])
        print(
            "ReCompGPT@w/o_diffusion_1 ALL: {}, L1: {}, L2: {}, L3: {},"
            "\nReCompGPT@w/o_diffusion_3 ALL: {}, L1: {}, L2: {}, L3: {}"
            .format(wo_diff_1['all'], wo_diff_1['l1'], wo_diff_1['l2'], wo_diff_1['l3'],
                    wo_diff_3['all'], wo_diff_3['l1'], wo_diff_3['l2'], wo_diff_3['l3']))
        print("")
        act_rel = eva_act_rel("results/rq1/records_regen@3_1")
        print("Valuable actions / All actions: {},\n"
              "Cases with valuable actions/ All cases: {},\n"
              "Correctly predicted cases / Cases with valuable actions:{},\n"
              "Cases with valuable actions / Correctly predicted cases :{}"
              .format(act_rel['m1'], act_rel['m2'], act_rel['m3'], act_rel['m4']))

        # average semantic similarity of diffused actions
        action_simi()

    if args.rq == "rq3":
        dm_1 = comt_avg(["results/rq3/records_regen@1_" + str(i+1) for i in range(3)], eva='dm', k=1)
        dm_3 = comt_avg(["results/rq3/records_regen@3_" + str(i+1) for i in range(3)], eva='dm')
        print("D-M@1 avg: {}, D-M@3 avg: {}".format(dm_1, dm_3))

    if args.rq == "rq4":
        analytical_mc = comt_avg(["results/rq4/records_regen_md@3_" + str(i+1) for i in range(3)])["all"]
        analytical_dm = comt_avg(["results/rq4/records_regen_md@3_" + str(i+1) for i in range(3)], eva='dm')
        print("analytical@3 filling rate, M-C: {}, D-M: {}".format(analytical_mc, analytical_dm))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", type=str, default='rq2', choices=['rq1', 'rq2', 'rq3', 'rq4'])
    args = parser.parse_args()
    main(args)
