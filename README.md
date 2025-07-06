## Overview

| file name        | description          |
| ---------------- | -------------------- |
| data             | dataset              |
| prompt           | prompts files        |
| records          | experimental results |
| evaluation.py    | evaluation script    |
| regen.py         | ReCompGPT main     |
| regen_local.py   | ReCompGPT local    |
| requirements.txt | dependency package   |




## Installation

Creating a Python Virtual Environment with Anaconda. Then clone and install the required libraries. 

```bash
conda create -n myenv python=3.11.0
conda activate myenv

git clone https://github.com/Jolouch/ReGen.git
cd ReGen
pip install -r requirements.txt
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
pip install en_core_web_lg-3.7.1-py3-none-any.whl
```


## Run
Before running, set openai api key. Enter your openai api key in `setting.ini`.

Run with openai api (gpt-4o).

```
python regen.py \
--cut_ratio 0.3 \ 
--cut_mtd random \
--is_diff True \
--generation 3 \
--rp records
```

Run with local llama3.3. First install the ollama client. Detail in [ollama/README.md at main · ollama/ollama · GitHub](https://github.com/ollama/ollama/blob/main/README.md#quickstart)

```
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3.3
```

Then run

```
python regen_local.py \
--model llama3.3 \
--generation 3 \
--rp records
```

##  Evaluation

We provide our experimental results for rq1-4.
Run for evaluation.

```
python evaluation.py --rq rq1  # rq2, rq3, rq4
```
The results of manual annotation are in the results file. Human assessment for llm evaluation is marked as 'human_eva_results', Action-relevance of RQ2 is marked as 'act_rel', and the D-M annotation is marked as 'D-M'. An example ([records_regen@3_1](results%2Frq1%2Frecords_regen%403_1)):
```
[
    {
        "desc": "model: gpt-4o, generation: 3, isDiffusion: True",
        "diff_act": [...],
        "act_rel": "0,0,1,0,1",
        "analysis": {...},
        "regen": [...],
        "label": "...",
        "absence": "...",
        "llm_eva_results": "1,1,1",
        "human_eva_results": "1,1,1",
        "D-M": "-1,-1,-1"
    }
]
```

## Prompts

The prompt of every task is list in the `prompt` directory.

| file name                 | usage                                                        |
| ------------------------- | ------------------------------------------------------------ |
| diffusion_usr_msg         | action diffusion                                             |
| extract_usr_msg           | extract actions from completed specifications                |
| filter_usr_msg            | filter actions that have been mentioned in original specifications |
| regen_sys_msg             | system prompt of ReCompGPT                                 |
| regen_usr_msg             | main prompt of ReCompGPT                                   |
| regen_usr_msg_vanilla     | vanilla prompt                                               |
| regen_usr_msg_vanilla_cot | vanilla CoT prompt                                           |

## Data samples

Data used in our paper is in the `data` directory. We list examples of three levels.

`Level 1 case`

| item                    | value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| sample_level            | 1                                                            |
| function_name           | Game Sequence                                                |
| function_description    | The component presents a series of multiple-choice fraction questions. Correct answers lead to next question. Incorrect answers prompt a retry without scoring. At critical points, plot direction changes based on the user's answer. After a set number of questions, the user is directed to the ending scene. |
| function_specifications | 1.This component will display a question, and then wait until the user chooses an answer.<br />2.If the incorrect answer is selected, this component will inform the user of this and give them another chance to answer the question, while their score will not count.<br />3.At certain critical points, this component will choose different directions in the plot based on whether the question at the critical point was answered correctly. 4.After the user has proceeded through a set number of questions, they will be directed to the ending scene component. |
| type                    | branch                                                       |
| absence                 | situation of selecting the correct answer                    |
| label                   | If the user selects the correct answer, a message to this effect will be displayed and the component will move to the next question. |

`Level 2 case`

| item                    | value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| sample_level            | 2                                                            |
| function_name           | Generate Unit Unavailable Event                              |
| function_description    | When a request from the thermostat for a heating unit or cooling to be turned is denied, an event should be generated for subsequent record |
| function_specifications | 1.When a request for a heating unit or cooling to be turned is denied, this procedure shall use the information about the thermostat and the heating or cooling unit to generate a specific system event.<br />2.This system event shall consist of a description of the event type (a request denied event) and a designation of the thermostat that made the request. |
| type                    | action(obj)                                                  |
| absence                 | designation of the not turned unit                           |
| label                   | This system event shall contain a designation of the heating or cooling unit that was not turned. |

`Level 3 case`

| item                    | value                                                        |
| ----------------------- | ------------------------------------------------------------ |
| sample_level            | 3                                                            |
| function_name           | Generate Alarm Data                                          |
| function_description    | There are two events that shall result in an alarm condition: 1) an invalid temperature value is reported from a thermostat, or 2) the reported temperature has exceeded the defined limits. This system shall record the alarm event. |
| function_specifications | 1. When the THEMAS system detects a request for an alarm, this process shall detect which of the two alarms are being requested. <br />2. If the system detects an invalid temperature, this process shall output a continuous series of alternating 500Hz and 700Hz beeps on the supervisor's computer. Each beep shall have a three-quarter second duration. <br />3. If the system detects a temperature limit has been exceeded, this process shall output a continuous series of alternating 1000Hz and 1500Hz beeps on the supervisor's computer. Each beep shall have a one-half second duration. <br />4. Each time an alarm is requested, an alarm event shall be recorded. This event shall be used to provide operational and statistical reports about the system. |
| type                    | action                                                       |
| absence                 | action of handling the alarm                                 |
| label                   | This series of beeps shall continue until the supervisor manually resets the alarm through the supervisor's interface window. |