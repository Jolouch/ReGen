import ast
import re
import time
from langchain_core.messages import AIMessage
import json
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from regen import save_record, cut_specs, diffusion, extract_actions_by_llm, filter_by_llm
from evaluation import eva_total, get_gen, simi, eva, accuracy


def record_process(record, generation, doc, fun, rp):
    def fmt(out: str, ii):
        parsed = ast.literal_eval(out)
        res = {'generation': ii, 'absent_element': parsed['absent element'], 'new_specification': parsed['new specification']}
        return res

    # analysis_record = record
    regen = [fmt(record['ai_message']['step4'], 1)]
    if generation > 1:
        regen += [fmt(record['ai_message']['step_replay_' + str(i + 2)]['step4'], i + 2) for i in range(generation - 1)]
    result_record = {
        "time": record['time'],
        "desc": record['desc'],
        "diff_act": record['diff_act'],
        "act_rel": "",
        "regen": regen,
    }
    del record['act_rel']
    del record['diff_act']
    save_record(doc, fun, record, rp=rp + '/' + '0_ai_message')
    save_record(doc, fun, result_record, rp=rp)


def load_steps(topic, fun_spec, action_sequence):
    step0 = open('prompt/prompt_steps/step0', 'r', encoding='utf-8').read()
    step1 = open('prompt/prompt_steps/step1', 'r', encoding='utf-8').read().format(topic=topic,
                                                                                   function_name=fun_spec["function_name"],
                                                                                   function_description=fun_spec[
                                                                                "function_description"])
    step1_reflect = open('prompt/prompt_steps/step1_reflect', 'r', encoding='utf-8').read()
    step2 = open('prompt/prompt_steps/step2', 'r', encoding='utf-8').read().format(
        function_specifications="\n".join(fun_spec["function_specifications"]))
    step2_reflect = open('prompt/prompt_steps/step2_reflect', 'r', encoding='utf-8').read()
    step3 = open('prompt/prompt_steps/step3', 'r', encoding='utf-8').read().format(
        operation_sequence="\n".join(action_sequence))
    step3_reflect = open('prompt/prompt_steps/step3_reflect', 'r', encoding='utf-8').read()
    step4 = open('prompt/prompt_steps/step4', 'r', encoding='utf-8').read()
    steps = [step0, step1, step1_reflect, step2, step2_reflect, step3, step3_reflect, step4]
    return steps


class RegenChatBot:
    def __init__(self, model='gpt-4o-mini', temperature=1, generation=1, isDiff=True):
        self.model = model
        self.temperature = temperature
        self.generation = generation
        self.isDiff = isDiff
        self.llm = ChatOpenAI(
            api_key="",
            model=self.model,
            temperature=self.temperature,
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    open('prompt/prompt_steps/regen_sys_msg', 'r', encoding='utf-8').read(),
                    # You are an excellent AI assistant
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.step_map = {
            "0": "step0",
            "1": "step1",
            "2": "step1_reflect",
            "3": "step2",
            "4": "step2_reflect",
            "5": "step3",
            "6": "step3_reflect",
            "7": "step4",
        }
        self.in_tokens = 0
        self.out_tokens = 0

    def bot(self):
        # Define the function that calls the model
        def call_model(state: MessagesState):
            chain = self.prompt | self.llm
            response = chain.invoke(state)
            return {"messages": response}

        # Define a new graph
        workflow = StateGraph(state_schema=MessagesState)
        # Define the (single) node in the graph
        workflow.add_edge(START, "model")
        workflow.add_node("model", call_model)
        # Add memory
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        return app

    def count_tokens(self, app, config, is_replay=False):
        res = app.get_state(config).values['messages'] if not is_replay else app.get_state(config).values['messages'][10:]
        for msg in res:
            if 'token_usage' in msg.response_metadata:
                self.in_tokens += msg.response_metadata['token_usage']['prompt_tokens']
                self.out_tokens += msg.response_metadata['token_usage']['completion_tokens']

    # single file 
    def __call__(self, doc, rp='records'):
        app = self.bot()
        with open("input_info/" + doc, "r", encoding="utf-8") as f:
            input_info = json.load(f)
        input_info["specifications"] = input_info["specifications"][0:1]  # partial
        ci = 0
        for fun_spec in input_info["specifications"]:
            print('-' * 30 + fun_spec["function_name"] + '-' * 30)
            fun_name = fun_spec["function_name"]
            if re.match(r'.*\d$', fun_name):  # same fucntion name file split
                fun_name = fun_name[:-1]
            if self.isDiff:
                prompt_info = (input_info["topic"], fun_name, fun_spec["function_description"])
                specs, truncated_specs = cut_specs(fun_spec["function_specifications"])
                diffusion_result = diffusion(specs, truncated_specs, prompt_info)
                action_sequence0 = extract_actions_by_llm(diffusion_result, prompt_info)
                action_sequence = filter_by_llm(action_sequence0, fun_spec["function_specifications"])
            else:
                action_sequence = []
            steps = load_steps(input_info["topic"], fun_spec, action_sequence)

            config = {"configurable": {"thread_id": "regen_doc_fun" + str(ci)}}
            msg_tmp = {}
            print("regen processing, llm calling...")
            for i, step in enumerate(steps):
                for event in app.stream({"messages": [("user", step)]}, config, stream_mode="values"):
                    if isinstance(event["messages"][-1], AIMessage):
                        # event["messages"][-1].pretty_print()
                        msg_tmp[self.step_map[str(i)]] = event["messages"][-1].content
                        # if i == len(steps) - 1:  # output for final step
                        #     print(event["messages"][-1].content)
            self.count_tokens(app, config)
            # pass@k
            if self.generation - 1 > 0:
                print("regen replay processing, llm calling...")
                for g in range(self.generation - 1):
                    to_replay = None
                    for state in app.get_state_history(config):
                        # print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
                        # print("-" * 30)
                        if len(state.values["messages"]) == 11:  # the third step
                            to_replay = state
                    # back to the third step
                    for event in app.stream(None, to_replay.config, stream_mode="values"):
                        if isinstance(event["messages"][-1], AIMessage):
                            # event["messages"][-1].pretty_print()
                            msg_tmp['step_replay_' + str(g + 2)] = {self.step_map['5']: event["messages"][-1].content}
                    # proceed after backtrack
                    for j, step in enumerate(steps[6:]):
                        for event in app.stream({"messages": [("user", step)]}, config, stream_mode="values"):
                            if isinstance(event["messages"][-1], AIMessage):
                                # event["messages"][-1].pretty_print()
                                msg_tmp['step_replay_' + str(g + 2)][self.step_map[str(6 + j)]] = event["messages"][-1].content
                    self.count_tokens(app, config, is_replay=True)
            record = {
                "time": time.ctime(),  
                "desc": "model: {}, temperature: {}, generation: {}, isDiffusion: {}".format(self.model,
                                                                                             self.temperature,
                                                                                             self.generation,
                                                                                             self.isDiff),
                "diff_act": action_sequence,
                "act_rel": "",  # 
                "ai_message": msg_tmp,
            }
            ci += 1
            record_process(record, self.generation, doc, fun_spec['function_name'], rp)

    def run_all(self):
        for doc in os.listdir("input_info"):
            self.__call__(doc)


if __name__ == '__main__':
    bot = RegenChatBot(model='gpt-4o', temperature=1, generation=1, isDiff=True)
    bot(doc='1999 - tcs.json', rp='records_bot')
    # regen token
    print(bot.in_tokens, bot.out_tokens)

    # eval
    # eva_total('records_bot', 'gpt-4o-mini')
    # single file eva
    # eva_data = get_gen("1998 - themas", rp='records_bot')
    # sm = simi(eva_data)
    # llm_sm = eva(eva_data, model='gpt-4o')
    # for j in range(len(sm)):
    #     print(llm_sm[j], sm[j])
    # print(accuracy(llm_sm))
