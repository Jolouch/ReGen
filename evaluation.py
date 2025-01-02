import argparse
import json
import os
import time
from collections import Counter
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
import configparser

config = configparser.ConfigParser()
config.read('setting.ini')
api_key = config['API']['api_key']


def simi(data):
    model = SentenceTransformer('all-mpnet-base-v2')
    result = []
    for fun, values in data.items():
        s1 = values["absence"]  # absence, label
        s2 = values["predictions"]

        embeddings1 = model.encode(s1)
        embeddings2 = model.encode(s2)

        similarities = model.similarity(embeddings1, embeddings2)
        # similarities = torch.round(similarities * 100) / 100
        result.append((fun, np.round(similarities.numpy().astype(np.float64), 2).tolist()[0]))
    return result


def get_gen(label_data, doc, rp='records'):
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
    # print(llm_eva_results)
    t = 0
    for item in llm_eva_results:
        if 'true' in item[1]:
            t += 1
    # print(t, len(llm_eva_results))
    return t, len(llm_eva_results)


def fmt_eva_res(eva_res_str):
    eva_res_str = eva_res_str.lower()
    return 'true' if 'true' in eva_res_str else 'false'


def eva(eva_data, model='gpt-4o', temperature=1):
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
    total_true = 0
    total_samp = 0
    with open("data/re_data.json", "r", encoding="utf-8") as f:
        re_data = json.load(f)
    for doc in os.listdir(rp):
        print("------eva doc: {}---------".format(doc))
        eva_data = get_gen(re_data[doc], doc, rp)
        sm = simi(eva_data)
        llm_sm = eva(eva_data, model=model)
        for i, res in enumerate(llm_sm):
            # print(llm_sm[i], sm[i])
            with open(rp + "/" + doc + "/" + res[0] + ".json", 'r', encoding='utf-8') as fr:
                regen_data = json.load(fr)
            llm_eva_results = [1 if item == 'true' else 0 for item in res[1]]
            regen_data[-1]["label"] = eva_data[res[0]]["label"]
            regen_data[-1]["absence"] = eva_data[res[0]]["absence"]
            regen_data[-1]['semantic_similarity'] = ",".join(map(str, sm[i][1]))
            regen_data[-1]['llm_eva_results'] = ",".join(map(str, llm_eva_results))
            regen_data[-1]['human_eva_results'] = ""
            with open(rp + "/" + doc + "/" + res[0] + ".json", "w", encoding="utf-8") as fw:
                json.dump(regen_data, fw, ensure_ascii=False, indent=4)
        total_true += accuracy(llm_sm)[0]
        total_samp += accuracy(llm_sm)[1]
    print("\n accuracy: {:.2f}%".format(total_true / total_samp * 100))


def acc_sample_levels(rp):
    from utils import statistics
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
    print("ALL: {:.2f}%".format(acc_total / total_samples * 100))
    for i in range(3):
        print("level {} accuracy: {:.2f}%".format(i + 1, acc_count[str(i + 1)] / count[str(i + 1)] * 100))


def eva_llm_human(rp):
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
    print("accuracy_score:{:.2f}%, precision_score:{:.2f}%, recall_score:{:.2f}%, f1_score:{:.2f}%".format(
        accuracy_score * 100,
        precision_score * 100,
        recall_score * 100,
        f1_score * 100))


def eva_act_rel(rp):
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
    print("valuable actions {:.2f}%".format(useful_actions / action * 100))
    print("cases with valuable actions: {:.2f}%".format(useful_actions_coverage / total_case * 100))
    print("correctly predicted cases in cases with valuable actions {:.2f}%".format(
        true_pred / useful_actions_coverage * 100))
    print("cases with valuable actions in correctly predicted cases {:.2f}%".format(act_rel / total_true_case * 100))


def eva_d_m(rp):
    json_files = []
    useless_gen = 0
    m_c = 0
    d_m = 0
    for root, _, files in os.walk(rp):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    total_gen = len(json_files) * 3
    for case in json_files:
        with open(case, 'r', encoding='utf-8') as f:
            regen_data = json.load(f)
        dm = list(map(int, regen_data[-1]["D-M"].split(",")))
        assert len(dm) == 3
        cnt = Counter(dm)
        useless_gen += cnt[0]
        m_c += cnt[-1]
        d_m += cnt[1]
    print("D-M(ex M-C) {:.2f}%".format(d_m / (total_gen - m_c) * 100))


if __name__ == '__main__':
    # eva total
    # eva_total("records", model="gpt-4o")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rp", type=str, default='records')
    args = parser.parse_args()
    rp = args.rp

    # eva level-wise
    print("eva total, level-wise:")
    acc_sample_levels(rp)
    print("")

    # eva llm results
    print("eva llm results:")
    eva_llm_human(rp)
    print("")

    # eva rel action
    print("eva rel action:")
    eva_act_rel(rp)
    print("")

    # eva D-M
    print("eva D-M:")
    eva_d_m(rp)
