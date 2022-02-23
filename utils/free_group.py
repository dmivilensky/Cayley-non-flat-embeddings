import torch
import numpy


def lcp(strs):
    if len(strs) == 0:
        return ""
    current = strs[0]
    for i in range(1, len(strs)):
        temp = ""
        if len(current) == 0:
            break
        for j in range(len(strs[i])):
            if j < len(current) and current[j] == strs[i][j]:
                temp += current[j]
            else:
                break
        current = temp
    return current


def pairwise_distances(sequences):
    with torch.no_grad():
        batch_size = sequences.shape[0]
        result = numpy.zeros(shape=(batch_size, batch_size))
        for i in range(batch_size):
            for j in range(i + 1):
                s1 = "".join(map(lambda x: str(x.item()).strip("0"), sequences[i]))
                s2 = "".join(map(lambda x: str(x.item()).strip("0"), sequences[j]))
                result[i, j] = result[j, i] =\
                    len(s1) + len(s2) - 2 * len(lcp([s1, s2]))
        return torch.Tensor(result)

'''
    Числа 2 * i - 1 и 2 * i для i > 0 являются обратными элементами свободной гурппы.
    Ноль - нейтратльный элемент
'''

def is_inversed(a, b):
    '''
        Метод проверяет правда ли что a * b = 1.
        a и b одинакового типа. Тип может быть либо int64, либо array 
    '''
    assert type(a) == type(b)
    if isinstance(a, numpy.ndarray):
        return numpy.array_equal(a, inverse(b))
    else:
        return a == inverse(b)


def inverse(a):
    '''
        Метод возвращает обратный для a
        у a тип может быть либо int64, либо array
    '''
    def int_impl(a):
        return 0 if a <= 0 else ((a - 1) ^ 1) + 1
    def list_impl(a):
        return numpy.vectorize(int_impl)(a[::-1])
    return list_impl(a) if isinstance(a, numpy.ndarray) else int_impl(a)


def remove_targets(word, targets):
    '''
        Метод убирает вхождение targets в word.
        targets либо 2d array либо 1d array.

        Метод помагает проверить, что элемент лежит в нормальной подгруппе образованной каким-нибудь элементом.
        word = ...
        target, inverse_target = [2 * i], [2 * i - 1]
        if reduce(remove_targets(word, [target, inverse_target])).size == 0:
            print(f'{word} \in <target>^F')
    '''
    def remove_target(word, target):
        n, m = word.shape[0], target.shape[0]
        mask, pointer = numpy.array([False] * n), 0
        while pointer < n:
            if numpy.array_equal(word[pointer:pointer + m], target):
                pointer += m
            else:
                mask[pointer] = True
                pointer += 1
        return mask
    mask = numpy.array([True] * word.shape[0])
    if len(targets.shape) == 1:
        mask &= remove_target(word, targets)
    else:
        for target in targets:
            mask &= remove_target(word, target)
    return word[mask]


def reduce(word):
    '''
        Метод приводит word в нормальную форму
    '''
    stack = []
    for w in word:
        if w == 0:
            continue
        if len(stack) == 0 or not is_inversed(stack[-1], w):
            stack.append(w)
        else:
            stack.pop()
    return numpy.array(stack)

def remove_elem_and_inverse(word, target):
    return remove_targets(word, numpy.array([target, inverse(target)]))

def reduce_and_pad(word, target_length):
    reduced_word = reduce(word)
    return numpy.pad(reduced_word, (0, target_length - len(reduced_word)), constant_values=(0, 0))
