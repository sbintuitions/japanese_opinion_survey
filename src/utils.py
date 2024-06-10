import itertools

def make_permutations(qa, all_permutations: bool = False) -> list[int]:
    choice_length = len(qa.choices)
    if all_permutations:
        permutations = list(itertools.permutations([i for i in range(choice_length)], choice_length))
    else:
        permutations = [[(i + j) % choice_length for j in range(choice_length)] for i in range(choice_length)]
    return permutations
