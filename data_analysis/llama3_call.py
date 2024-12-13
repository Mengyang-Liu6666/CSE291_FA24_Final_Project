import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import torch
import time

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

def llama3_pred(query, do_sample = False):
    inputs = tokenizer(query, return_tensors="pt").to("cuda")
    prompt_length = len(query)
    with torch.no_grad():
        outputs = model.generate(**inputs, 
                                 pad_token_id=tokenizer.eos_token_id, 
                                 max_new_tokens = 1000, 
                                 do_sample = do_sample,
                                 stopping_criteria=create_stop_criteria("<|im_end|>", query))
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response[prompt_length:]

def main():
    query = ""
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

    
    # Test question 1
    # question = "If $f(x) = \\frac{3x-2}{x-2}$, what is the value of $f(-2) +f(-1)+f(0)$?"
    # action_1 = "We need to first find $f(2)+f(1)+f(0)$"
    # action_2 = "We need to first find $f(-2)+f(-1)+f(0)$"

    # Test question 2
    question = "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \le \\theta < 2 \pi.$"
    action_1 = "To convert the point \\((0,3)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\). The formulas for converting from rectangular coordinates \\((x, y)\\) to polar coordinates \\((r, \\theta)\\) are:"
    action_2 = "To convert the point \\((0,2)\\) from rectangular coordinates to polar coordinates, we need to find the values of \\(r\\) and \\(\\theta\\)."

    # Prompts
    prompt = "<|im_start|>user\nHere is a (partially completed) Math question and two potential next steps.\n"
    prompt += "Question: \"{}\"\nPotential step 1: \"{}\"\nPotential step 2: \"{}\"\n"
    prompt += "Answer 'Yes' if these two steps mathematically same or one step extends the other. Otherwise, answer 'No'. Give a brief reasoning before your answer.<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate few shot examples
    few_shot_examples = ""
    for i in range(len(question_ex)):
        example = prompt.format(question_ex[i], action_1_ex[i], action_2_ex[i])
        example += f"{answer_ex[i]}<|im_end|>\n\n"
        few_shot_examples += example
    
    query = few_shot_examples + prompt.format(question, action_1, action_2)

    print(query)
    total = 50
    yes_count = 0
    no_count = 0

    start_time = time.time()
    for i in range(total):
        pred = llama3_pred(query, do_sample=True)
        print(pred[:pred.index("<|im_end|>")])
        if "Yes" in pred:
            yes_count += 1
        elif "No" in pred:
            no_count += 1

    end_time = time.time()
    time_spent = end_time - start_time
    print(f"\n Total: {total}, Yes: {yes_count/total*100}%, No: {no_count/total*100}%, Invalid: {(total-yes_count-no_count)/total*100}%, seconds per query: {time_spent/total}")

if __name__ == "__main__":
    main()