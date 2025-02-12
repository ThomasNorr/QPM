def index_list_with_sorting(list_to_sort, sorting_list):
    answer = []
    for entry in sorting_list:
        answer.append(list_to_sort[entry])
    return answer


def mask_list(list_input, mask):
    return [x for i, x in enumerate(list_input) if mask[i]]


def txt_load(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

