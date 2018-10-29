# -*-coding=utf-8-*-
import Levenshtein as Lev


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
    # print Lev.distance(s1, s2)
    print(Lev.distance(s1, s2))
    return Lev.distance(s1, s2)


def cer_s(data):
    count = 0
    size = 0
    for pair in data:
        index = Lev.distance(pair[1].strip(), pair[0].strip())
        print(pair[1], pair[0], index, len(pair[1].strip()), len(pair[0].strip()))
        count += Lev.distance(pair[1].strip(), pair[0].strip())
        size += len(pair[1].strip())
    print(size, count, float(count) / size)
    return count, float(count) / size
