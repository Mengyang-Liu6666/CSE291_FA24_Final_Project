import json
import re
import jsonlines
import argparse
import os

COT_TASK_DESC = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>"
PROBLEM_FORMAT_STR = """<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"""

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
step_tag = "\n\n\n\n\n"

# Utility method
def get_prompt(question):
    prompt = [DEFAULT_BOS_TOKEN, COT_TASK_DESC, "\n", PROBLEM_FORMAT_STR.format(question=question) ,"\n"]
    return "".join(prompt)

class Node():
    def __init__(self, path_idx=-1, text="", value=2, rest_value=2, depth = 0, children=None) -> None:
        self.path_idx_list = [path_idx]
        self.text = text
        self.value = value
        self.rest_value = rest_value
        self.depth = depth
        self.children = children
        if not children:
            self.children = []
            
        
        return None
    
    def append_node(self, node) -> None:
        self.path_idx_list.extend(node.path_idx_list)
        for self_child_node in self.children:
            if self_child_node.text == node.text:
                # Sharing a state
                self_child_node.rest_value = min(self.rest_value, node.rest_value)
                for node_children in node.children:
                    self_child_node.append_node(node_children)
                    return None
                
        # Not sharing a state
        self.children.append(node)
        return None
    
    def to_dict(self):
        node_dict = {}
        node_dict["path_idx_list"] = self.path_idx_list
        node_dict["text"] = self.text
        node_dict["value"] = self.value
        node_dict["rest_value"] = self.rest_value
        node_dict["depth"] = self.depth
        node_dict["children"] = [self_child_node.to_dict() for self_child_node in self.children]
        return node_dict

def traj_to_node(traj) -> Node:
    path_idx = traj["path_idx"]
    text_list = traj["text"].split("\n\n")
    value_list = traj["value"]
    rest_value_list = traj["rest-mcts value"]

    children = []
    for i in range(len(value_list)-1, -1, -1):
        cur_node = Node(path_idx=path_idx, 
                        text=text_list[i],
                        value=value_list[i],
                        rest_value=rest_value_list[i],
                        depth=i+1,
                        children=children)
        children = [cur_node]

    return cur_node

def convert(solution):
    data = {}
    data["question"] = solution["question"]
    data["groundtruth"] = solution["groundtruth"]
    data["result"] = solution["result"]

    root_node = Node() # Trivial node

    for traj in solution["output"]:
        traj_node = traj_to_node(traj)
        root_node.append_node(traj_node)

    root_node_dict = root_node.to_dict()

    data["solution"] = root_node_dict["children"]

    return data

def main():

    parser = argparse.ArgumentParser(description="A script that convert record.jsonl to trajectory form.")
    parser.add_argument("i", type=str, help="Input file name")
    parser.add_argument("--o", type=str, help="Output file name")

    args = parser.parse_args()

    # Pre-process args
    input_name = args.i
    if input_name.find(".jsonl") == -1:
        input_name = f"{input_name}.jsonl"
    if args.o:
        output_name = args.o
        if output_name.find(".jsonl") == -1:
            output_name = f"{output_name}.jsonl"
    else:
        index = input_name.find(".jsonl")
        output_name = f"{input_name[:index]}_converted.jsonl"

    with open(input_name, 'r', encoding='utf-8') as json_file:
        solutions = [json.loads(line) for line in json_file]

    data_list = []
    total = len(solutions)
    for i, solution in enumerate(solutions):
        data_list.append(convert(solution))
        if (i+1) % 200 == 0:
            print(f"Converted {i+1}/{total} problems.")
    with jsonlines.open(output_name, mode='w') as writer:
        writer.write_all(data_list)

if __name__ == "__main__":
    main()