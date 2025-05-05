from argparse import ArgumentError
from tqdm import tqdm
import random
from typing import Literal

from algorithms.prompt_leakage.judger.ai_judger import AIJudger
from algorithms.prompt_leakage.judger.ijudger import Judger
from algorithms.prompt_leakage.judger.regex_judger import RegexJudger
from algorithms.prompt_leakage.utils import call_llm_general, load_model, load_domain_examples


def run_no_defenses(target_model: str,
        prompt_file: str,
        attack_prompt_file: str,
        follow_up_file: str,
        domains: list[str],
        iterations: int,
        judger_type: Literal["gen_ia", "regex"]) -> list:

    base_prompt_file = "algorithms/prompt_leakage/prompts/"
    base_attack_dir = f"{base_prompt_file}attack_vectors/"

    attack_prompts = open(f"{base_attack_dir}{attack_prompt_file}", 'r').readlines()

    prompt_file = f"{base_prompt_file}{prompt_file}"

    follow_up_prompt_file = f"{base_prompt_file}{follow_up_file}"
    follow_up_prompt = open(follow_up_prompt_file, 'r').read()

    judger: Judger
    if judger_type == "gen_ia":
        judgement_prompt = open("algorithms/prompt_leakage/prompts/judge/judge-basic.txt", 'r').read()
        judger = AIJudger("llama3:8b", judgement_prompt)
    else:
        judger = RegexJudger()

    llm = load_model(target_model)     # loads the appropriate model, will

    n_per_domain: int
    if iterations % len(domains) != 0:
        raise ArgumentError(None, f"Iterations cannot be distributed equally between the domains. Actual value: {iterations}")

    n_per_domain = int(iterations / len(domains))

    results = []

    for domain in domains:

        llm_prompt = open(prompt_file.format(domain), 'r').read()

        dataset, ic_examples = load_domain_examples(domain)
        iter_dataset = dataset[:n_per_domain]

        for item in tqdm(iter_dataset, desc=f"Running Prompt Leakage No Defenses - Domain {domain}"):

            usr_inp = item['query']
            doc1 = item['doc1']
            doc2 = item['doc2']

            attack_vector = random.choice(attack_prompts)

            llm_input1 = llm_prompt.replace("<DOCUMENT>",usr_inp).replace("<RETRIEVED_DOC_1>",doc1).replace("<RETRIEVED_DOC_2>",doc2).replace("<ATTACK_VECTOR>", attack_vector)

            messages = [
                    {"role": "user", "content": llm_input1},
                ]

            # First call to the LLM
            llm_response1 = call_llm_general(llm, messages)

            messages = [
                    {"role": "user", "content": llm_input1},
                    {"role": "assistant", "content": llm_response1},
                    {"role": "user", "content": follow_up_prompt},
                ]

            # Subsequent call to the LLM
            llm_response2 = call_llm_general(llm, messages)

            result = {
                "method": "No Defenses",
                "setup_message": None,
                "first_message": llm_input1,
                "second_message": follow_up_prompt,
                "first_response": llm_response1,
                "second_response": llm_response2
            }

            result = judger.judge(result, doc1, doc2)

            results.append(result)

    return results