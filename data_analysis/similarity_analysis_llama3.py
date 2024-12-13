import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import jsonlines
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=True,
    device_map="auto",
    torch_dtype=torch.float16
)

question_ex = [
    "Solve for $x$ and $y$: $x+y=5$, $x-y=1$",
    "Find the local maximum and minimum points for the function $h(x)=x^3-3x^2+4$",
    "Let $X$ and $Y$ be discrete random variables. Suppose we have access to $\\forall x, y, P(X=x,Y=y)$. How to find $P(X=x)$?",
    "We have 10 chicken and rabbits. Each chicken has 2 legs and each rabbit has 4 legs. In total, they have 32 legs. How many chickens do we have?",
    # "Prove the following inequality: $\\log(x)\\le x-1$",
    "A rectangle has width $x$. The length is $2x$. What is the area of the rectangle?",
    "Find the sum of a geometric series with the first term to be $4$ and the ratio to be $0.2$",
]
action_1_ex = [
    "Express $x$ in terms of $y$, we have: $x=5-y$",
    "To find the local maximum and minimum points, we can first find $h'(x)$",
    "We can apply the rule of marginalization on $Y$.",
    "Let the number of chickens be $x$, then we have $10-x$ rabbits.",
    # "To find prove the inequality, we first move all terms to the left, which gives $\\log(x)+1-x\\le 0$",
    "To find the area of the rectangle, we can first find the length, which is $3x$.",
    "By substitution to the formula for the sum of an infinite geometric series, we have $4/(1-0.2)=5$",
]
action_2_ex = [
    "To find $x$ and $y$, we can first $y$ in terms of $x$, we have: $y=5-x$",
    "To find the local maximum and minimum points, we can first differentiate the original function.",
    "We can sum over all possible $y$ for each $x$.",
    "Assume all 10 animals are chickens, then in total they have $2\\times 10=20$ legs.",
    # "To find prove the inequality, we can move all terms to the left hand side, which gives $\\log(x)+1-x\\le 0$. Then we can define this part to be $g(x)=\\log(x)+1-x$ and differentiate it: $g'(x)=1/x-1$.",
    "The width is $x$, therefore the length is $3x$.",
    "The formula for the sum of an infinite geometric series is $S=a/(1-r)$, where $a$ is the first term and $r$ is the common ratio. Therefore, we have $S=4/(1-0.2)=10$"
]
answer_ex = [
    "(starting with different variables: $x$ and $y$, thus different) No",
    "(same step, including the question content $h(x)$ or not, thus the same) Yes",
    "(same step, different ways to call the same idea: rule of marginalization is summing over an variable, thus the same) Yes",
    "(using reasoning or equations, thus different) No",
    # "(one step extends the other, thus the same) Yes", 
    "(same step, one is just a rephrase of the other) Yes",
    "(same step, but different answers: $5$ or $10$, thus different) No",
]

# Prompts
prompt = "<|im_start|>user\nHere is a (partially completed) Math question and two potential next steps.\n"
prompt += "Question: \"{}\"\nPotential step 1: \"{}\"\nPotential step 2: \"{}\"\n"
prompt += "Answer 'Yes' if these two steps mathematically same or one step extends the other. Otherwise, answer 'No'. Give a brief reasoning before your answer.<|im_end|>\n<|im_start|>assistant\n"



class StringStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_string, tokenizer, prompt):
        self.stop_string = stop_string
        self.tokenizer = tokenizer
        self.prompt_length = len(prompt)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded_text[self.prompt_length:]

def create_stop_criteria(stop_string, prompt):
    string_stopping= StringStoppingCriteria(stop_string=stop_string, 
                                            tokenizer=tokenizer,
                                            prompt=prompt)
    stopping_criteria = StoppingCriteriaList([string_stopping])
    return stopping_criteria

def llama3_pred(question, action_1, action_2, few_shot_examples, do_sample = False):
    query = few_shot_examples + prompt.format(question, action_1, action_2)
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    prompt_length = len(query)
    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                 pad_token_id=tokenizer.eos_token_id, 
                                 max_new_tokens = 1000, 
                                 do_sample = do_sample,
                                 stopping_criteria=create_stop_criteria("<|im_end|>", query))
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "No" in response[prompt_length:]:
        return 0
    else:
        return 1

############ Specify input filename here ############

input_name = "test_5_legal_actions.jsonl"

def main():
    # Prepare few-shot_examples
    few_shot_examples = ""
    for i in range(len(question_ex)):
        example = prompt.format(question_ex[i], action_1_ex[i], action_2_ex[i])
        example += f"{answer_ex[i]}<|im_end|>\n\n"
        few_shot_examples += example

    # Read log
    with open(input_name, 'r', encoding='utf-8') as json_file:
        solutions = []
        for i, line in enumerate(json_file):
            if i % 2 == 0:
                solutions.append(json.loads(line.strip()))

    # Organize data
    similarity_data = {}
    i = 0
    for solution_group in solutions:
        for sol in solution_group:
            if sol["question"] not in similarity_data.keys():
                similarity_data[sol["question"]] = {"question_id": i, "partial_sol": {}}
                i += 1
        
            previous_steps = sol["input_prefix"].split("<|im_end|>\n<|im_start|>assistant\n")[1]

            previous_step_list = previous_steps.split("\n\n")[:-1]
            if not len(previous_step_list)==(sol["depth"]-1):
                print("Error: number of steps not matching depth.")
                print(f"Step number: {len(previous_step_list)}")
                depth = sol["depth"]
                print(f"Depth: {depth-1}")

            if len(previous_steps) >= 2:
                previous_steps = previous_steps[:-2]

            if previous_steps not in similarity_data[sol["question"]]["partial_sol"].keys():
                similarity_data[sol["question"]]["partial_sol"][previous_steps] = {}
                similarity_data[sol["question"]]["partial_sol"][previous_steps]["action_list"] = set()
                similarity_data[sol["question"]]["partial_sol"][previous_steps]["cluster_list"] = [[]] # place-holder
                similarity_data[sol["question"]]["partial_sol"][previous_steps]["adjacency_mat"] = [[]] # place-holder

            similarity_data[sol["question"]]["partial_sol"][previous_steps]["action_list"].add(sol["action"].replace("\n\n", ""))

    new_sim_data = []

    for question_text, question_prop in similarity_data.items():
        partial_sol_list = []
        for partial_sol_text, partial_sol_dict in question_prop["partial_sol"].items():
            partial_sol_prop = {}
            partial_sol_prop["partial_sol_text"] = partial_sol_text
            partial_sol_prop["action_list"] = list(partial_sol_dict["action_list"])
            partial_sol_prop["cluster_list"] = partial_sol_dict["cluster_list"]
            partial_sol_prop["adjacency_mat"] = partial_sol_dict["adjacency_mat"]
            partial_sol_list.append(partial_sol_prop)
        new_sim_data.append({"question": question_text, "partial_sol": partial_sol_list})
        
    similarity_data = {}

    # Compute similarities
    for question_data in new_sim_data:
        for partial_sol_dict in question_data["partial_sol"]:
            question_text = question_data["question"]
            partial_sol_text = partial_sol_dict["partial_sol_text"]
            combined_question = f"{question_text}\n{partial_sol_text}"
            action_list = partial_sol_dict["action_list"]
            adjacency_mat = [[0 for _ in range(len(action_list))] for _ in range(len(action_list))]
            cluster_list = [[i] for i in range(len(action_list))]
            for i in range(len(action_list)):
                for j in range(i+1, len(action_list)):
                    action_1 = action_list[i]
                    action_2 = action_list[j]

                    deberta_prediction = llama3_pred(combined_question, 
                                                     action_1, 
                                                     action_2, 
                                                     few_shot_examples,
                                                     do_sample=False)

                    if deberta_prediction == 1:
                        adjacency_mat[i][j] = 1
                        adjacency_mat[j][i] = 1
                        for cluster in cluster_list:
                            if i in cluster and j not in cluster:
                                # Need merge
                                for cluster_2 in cluster_list:
                                    if j in cluster_2:
                                        cluster.extend(cluster_2)
                                        cluster_list.remove(cluster_2)
                                        break
                        if [j] in cluster_list:
                            cluster_list.remove([j])
                
            for cluster in cluster_list:
                cluster.sort()
            partial_sol_dict["adjacency_mat"] = adjacency_mat
            partial_sol_dict["cluster_list"] = cluster_list

    ############ Specify output filename here ############

    output_name = "test_5_legal_actions_similarity_llama3.json"
    with jsonlines.open(output_name, mode='w') as writer:
        writer.write_all(new_sim_data)

if __name__ == "__main__":
    main()