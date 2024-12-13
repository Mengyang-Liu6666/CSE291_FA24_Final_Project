import json
import re
import jsonlines

filename = "./gsm8k_test.jsonl"
output_name = "./MATH_gsm8k_test.jsonl"

with open(filename, 'r') as json_file:
    gsm8k = [json.loads(line) for line in json_file]

def replace_calculations(text):
    answer = text.split('####')[-1].strip()
    text = text.split('\n####')[0]
    parts = text.split(" ")
    ans_found = False
    for i in range(len(parts)-1, -1, -1):
        sub_parts = parts[i].split("\n")
        for j in range(len(sub_parts)):
            sub_parts[j]
            result = re.search(r'<<(.*?)>>', sub_parts[j])
            if result:
                formula = result.group(1)
                # Handle multiplication and division
                formula = formula.replace("*", "\\times ")
                formula = formula.replace("/", "\\div ")
                if not ans_found and formula.endswith(answer):
                    formula = formula.replace(answer, f"\\boxed{{{answer}}}")
                    ans_found = True
                sub_parts[j] = f'${formula}$'
        parts[i] = "\n".join(sub_parts)
    text = " ".join(parts)

    # spaced_formula = re.sub(r'([\+\-\*=])', r' \1 ', formula)
    # spaced_formula = re.sub(r'\s+', ' ', spaced_formula).strip()
    # spaced_formula = re.sub(r'\s\d+$', '', spaced_formula)
    # text.replace(spaced_formula, "")

    return text, answer

def main():
    math_format_list = []
    for index, problem in enumerate(gsm8k):
        math_format = {}
        math_format["problem"] = problem["question"]
        math_format["level"] = "Level 1"

        solution, answer = replace_calculations(problem["answer"])
        math_format["solution"] = solution
        math_format["subject"] = "GSM8K"
        math_format["unique_id"] = index
        math_format["answer"] = answer

        math_format_list.append(math_format)

    with jsonlines.open(output_name, mode='w') as writer:
        writer.write_all(math_format_list)

if __name__ == "__main__":
    main()