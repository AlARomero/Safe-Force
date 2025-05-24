import random
import numpy as np

from graph.prompt_node import PromptNode


class SelectPolicy:
    def select(self, prompt_nodes: list[PromptNode]) -> PromptNode:
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes_to_update: 'list[PromptNode]', all_prompt_nodes_list: list[PromptNode]):
        pass


class RoundRobinSelectPolicy(SelectPolicy):
    def __init__(self):
        super().__init__()
        self.index: int = 0

    def select(self, prompt_nodes: list[PromptNode]) -> PromptNode:
        seed = prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes_to_update: 'list[PromptNode]', all_prompt_nodes_list: list[PromptNode]):
        self.index = (self.index - 1 + len(all_prompt_nodes_list)
                      ) % len(all_prompt_nodes_list)

class RandomSelectPolicy(SelectPolicy):
    def __init__(self):
        super().__init__()

    def select(self, prompt_nodes: list[PromptNode]) -> PromptNode:
        seed = random.choice(prompt_nodes)
        seed.visited_num += 1
        return seed

class UCBSelectPolicy(SelectPolicy):
    def __init__(self, prompt_nodes: list[PromptNode], explore_coeff: float = 1.0):
        super().__init__()

        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        self.rewards = [0 for _ in range(len(prompt_nodes))]

    def select(self, prompt_nodes: list[PromptNode]) -> PromptNode:
        if len(prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(prompt_nodes) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(prompt_nodes))
        for i, prompt_node in enumerate(prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            scores[i] = self.rewards[i] / smooth_visited_num + \
                self.explore_coeff * \
                np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = np.argmax(scores)
        prompt_nodes[self.last_choice_index].visited_num += 1
        return prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes_to_update: 'list[PromptNode]', all_prompt_nodes_list: list[PromptNode]):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes_to_update])
        self.rewards[self.last_choice_index] += \
            succ_num / len(all_prompt_nodes_list)

class MCTSExploreSelectPolicy(SelectPolicy):
    def __init__(self, questions: list[str], initial_prompt_nodes: list[PromptNode], ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__()
        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.questions = questions
        self.initial_prompts_nodes = initial_prompt_nodes.copy()
        self.ratio = ratio  # balance between exploration and exploitation
        self.alpha = alpha  # penalty for level
        self.beta = beta   # minimal reward after penalty

    def select(self, prompt_nodes: list[PromptNode]) -> PromptNode:
        self.step += 1
        max_index = max(pn.index for pn in self.initial_prompts_nodes)
        if len(self.rewards) <= max_index:
            self.rewards.extend([0] * (max_index + 1 - len(self.rewards)))
        if len(prompt_nodes) > len(self.rewards):
            self.rewards.extend([0 for _ in range(len(prompt_nodes) - len(self.rewards))])
        self.mctc_select_path.clear()
        cur = max(
            self.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) / (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)
        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)
        for pn in self.mctc_select_path:
            pn.visited_num += 1
        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes_to_update: 'list[PromptNode]', all_prompt_nodes_list: list[PromptNode]):
        index_to_pos = {pn.index: pos for pos, pn in enumerate(all_prompt_nodes_list)}
        succ_num = sum([prompt_node.num_jailbreak for prompt_node in prompt_nodes_to_update])
        pos = index_to_pos.get(self.last_choice_index)
        if pos is None:
            return
        last_choice_node = all_prompt_nodes_list[pos]
        #last_choice_node = all_prompt_nodes_list[self.last_choice_index]
        max_idx = max(pn.index for pn in self.mctc_select_path + [last_choice_node])
        if len(self.rewards) <= max_idx:
            self.rewards.extend([0] * (max_idx + 1 - len(self.rewards)))
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / (len(self.questions) * len(prompt_nodes_to_update))
            self.rewards[prompt_node.index] += reward * max(self.beta, (1 - 0.1 * last_choice_node.level))

class EXP3SelectPolicy(SelectPolicy):
    def __init__(self,
                 prompt_nodes: list[PromptNode],
                 gamma: float = 0.05,
                 alpha: float = 25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.last_choice_index = None
        self.weights = [1. for _ in range(len(prompt_nodes))]
        self.probs = [0. for _ in range(len(prompt_nodes))]

    def select(self, prompt_nodes: list[PromptNode]) -> PromptNode:
        # TODO Y si es menor?
        if len(prompt_nodes) > len(self.weights):
            self.weights.extend([1. for _ in range(len(prompt_nodes) - len(self.weights))])
        if len(prompt_nodes) > len(self.probs):
            self.probs.extend([0. for _ in range(len(prompt_nodes) - len(self.probs))])
        np_weights = np.array(self.weights)
        probs = (1 - self.gamma) * np_weights / np_weights.sum() + \
            self.gamma / len(prompt_nodes)
        self.last_choice_index = np.random.choice(
            len(prompt_nodes), p=probs)
        prompt_nodes[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]
        return prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes_to_update: 'list[PromptNode]', all_prompt_nodes_list: list[PromptNode]):
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes_to_update])
        r = 1 - succ_num / len(prompt_nodes_to_update)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(
            self.alpha * x / len(all_prompt_nodes_list))