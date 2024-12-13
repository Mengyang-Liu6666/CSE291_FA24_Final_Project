import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json
import jsonlines
import torch
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cuda()

def deberta_pred(text_1, text_2):
    input = text_1 + ' [SEP] ' + text_2
    encoded_input = tokenizer.encode(input, padding=True)
    prediction = model(torch.tensor(torch.tensor([encoded_input]), device='cuda'))['logits']
    predicted_label = torch.argmax(prediction, dim=1)

    reverse_input = text_2 + ' [SEP] ' + text_1
    encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
    reverse_prediction = model(torch.tensor(torch.tensor([encoded_reverse_input]), device='cuda'))['logits']
    reverse_predicted_label = torch.argmax(reverse_prediction, dim=1)

    deberta_prediction = 1
    if 0 in predicted_label or 0 in reverse_predicted_label:
        deberta_prediction = 0

    return deberta_prediction

############ Specify input filename here ############

input_name = "test_5_legal_actions.jsonl"

def main():
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

                    qa_1 = f"{combined_question}\n{action_1}"
                    qa_2 = f"{combined_question}\n{action_2}"

                    deberta_prediction = deberta_pred(qa_1, qa_2)

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

    output_name = "test_5_legal_actions_similarity_deberta.json"
    with jsonlines.open(output_name, mode='w') as writer:
        writer.write_all(new_sim_data)

if __name__ == "__main__":
    main()