def make_permutations(qa) -> list[int]:
    choice_length = len(qa.choices)
    permutations = [[(i + j) % choice_length for j in range(choice_length)] for i in range(choice_length)]
    return permutations
