import argparse
import re
import math
import requests
import json
from typing import Dict, Any, Optional, List

from evaluation import eva_total
from regen import cut_specs, save_record


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client

        :param base_url: Ollama Server address, default as http://localhost:11434
        """
        self.base_url = base_url.rstrip('/')

    def chat(
            self,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = False,
            format_schema: Optional[Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Send chat request to Ollama API

        :param model: model to be used
        :param messages: Message list, each message is a dictionary containing roles and contents
        :param stream: Whether to use streaming response, default is False
        :param format_schema: JSON Schema
        :param options: model options, e.g. temperature
        :param kwargs: other parameters
        :return: response data
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **kwargs
        }
        if format_schema is not None:
            payload["format"] = format_schema
        if options is not None:
            payload["options"] = options
        headers = {
            "Content-Type": "application/json"
        }

        # Send request
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # Check the response status
        response.raise_for_status()
        # Extract and parse content
        content_str = response.json().get("message", {}).get("content", "{}")

        # parse json data from response data
        def parse(text):
            pattern = r'```json\n(.*?)\n```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)

        try:
            parse_res = parse(content_str)
            return parse_res
        except json.JSONDecodeError as e:
            raise ValueError(f"Unable to parse message.content to JSON: {content_str}") from e


CUT_RATIO = 0.3


def diffusion_local(specs, truncated_specs, prompt_info, client):
    """action diffusion (local)

    @param specs: requirements specifications
    @param truncated_specs: truncated specifications
    @param prompt_info: prompt information
    @param client: ollama client
    @return: completed specifications
    """
    batch_len = len(specs[0])
    cut_len = math.ceil(CUT_RATIO * batch_len)
    topic, fun_name, fun_desc = prompt_info
    sys_msg = open("prompt/local/diffusion_sys_msg", encoding='utf8').read()
    usr_msg = open("prompt/local/diffusion_usr_msg", encoding='utf8').read()
    batch_info = [
        {"n": batch_len,
         "cut_len": cut_len,
         "topic": topic,
         "function_name": fun_name,
         "function_description": fun_desc,
         "specs": "\n".join(specs[i]),
         "truncated": "\n".join(truncated_specs[i])}
        for i in range(len(specs))]
    responses = []
    for batch in batch_info:
        usr_msg_format = usr_msg.format(**batch)
        response = client.chat(
            model="llama3.3",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg_format}],
            stream=False,
            format_schema=None,
            options={"temperature": 1}
        )
        responses.append(response)

    diffusion_result = [res['specifications'] if res is not None else [] for res in responses]
    return diffusion_result


def extract_actions_by_llm_local(specs, client):
    """extract actions from diffused specifications (local)

    @param specs: requirements specifications after diffusion
    @param client: ollama client
    @return: extracted actions
    """
    sys_msg = open("prompt/local/extract_sys_msg", encoding='utf8').read()
    usr_msg = open("prompt/local/extract_usr_msg", encoding='utf8').read()
    batch_info = [{"specifications": "\n".join(s)} for s in specs]
    responses = []
    for batch in batch_info:
        usr_msg_format = usr_msg.format(**batch)
        response = client.chat(
            model="llama3.3",
            messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg_format}],
            options={"temperature": 1}
        )
        responses.append(response)

    extracted_actions = [a for res in responses if res is not None for a in res['actions']]
    extracted_actions = [str(i + 1) + '.' + item for i, item in enumerate(extracted_actions)]
    return extracted_actions


def filter_by_llm_local(extracted_actions, fun_specs, client):
    """filter extracted actions (local)

    @param extracted_actions: extracted actions
    @param fun_specs: functions specifications
    @param client: ollama client
    @return: filtered actions
    """
    sys_msg = open("prompt/local/extract_sys_msg", encoding='utf8').read()
    usr_msg = open("prompt/local/filter_usr_msg", encoding='utf8').read()
    info = {"actions": "\n".join(extracted_actions),
            "specifications": "\n".join(fun_specs)}
    usr_msg_format = usr_msg.format(**info)
    response = client.chat(
        model="llama3.3",
        messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg_format}],
        options={"temperature": 1}
    )
    filtered_actions = response['actions'] if response is not None else []
    filtered_actions = [str(i + 1) + '.' + item for i, item in enumerate(filtered_actions)]
    return filtered_actions


def regen_local(args):
    generation = args.generation
    rp = args.rp
    model = args.model

    client = OllamaClient()

    with open("data/re_data.json", "r", encoding="utf-8") as f:
        input_info = json.load(f)
    for doc, cases in input_info.items():
        for fun_spec in cases["specifications"]:
            fun_name = fun_spec["function_name"]
            if re.match(r'.*\d$', fun_name):
                fun_name = fun_name[:-1]
            prompt_info = (cases["topic"], fun_name, fun_spec["function_description"])
            specs, truncated_specs = cut_specs(fun_spec["function_specifications"])
            diffusion_result = diffusion_local(specs, truncated_specs, prompt_info, client)
            action_sequence0 = extract_actions_by_llm_local(diffusion_result, client)
            action_sequence = filter_by_llm_local(action_sequence0, fun_spec["function_specifications"], client)

            sys_msg = open("prompt/local/regen_sys_msg", encoding='utf8').read()
            usr_msg = open("prompt/local/regen_usr_msg", encoding='utf8').read()
            batch_info = [{"topic": cases["topic"],
                           "function_name": fun_name,
                           "function_description": fun_spec["function_description"],
                           "function_specifications": "\n".join(fun_spec["function_specifications"]),
                           "operation_sequence": "\n".join(action_sequence)
                           } for _ in range(generation)]
            responses = []
            for batch in batch_info:
                usr_msg_format = usr_msg.format(**batch)
                response = client.chat(
                    model=model,
                    messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg_format}],
                    options={"temperature": 1}
                )
                responses.append(response)
            # record and print
            print("------" + fun_spec["function_name"] + "------")
            record_info = {
                "desc": "model: {}, generation: {}".format("llama3.3:70B Q4_K_M", generation),
                "diff_act": action_sequence,
                "regen": [],
            }
            for i, res in enumerate(responses):
                ans = {
                    "generation": str(i + 1),
                    "absent_element": res["absent_element"],
                    "new_specification": res["new_specification"]
                }
                record_info["regen"].append(ans)
                print("Final answer {}:".format(str(i + 1)))
                print("absent_element: ", res['absent_element'])
                print("new_specification: ", res['new_specification'])
            save_record(doc, fun_spec["function_name"], record_info, rp=rp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama3.3")
    parser.add_argument("--generation", type=int, default=3)
    parser.add_argument("--rp", type=str, default='records_local')

    args = parser.parse_args()

    regen_local(args=args)
    eva_total(rp=args.rp)
