import numpy as np


def jelinek_mercer_smoothing(tf, lamda):
    """

    :param tf: Term-Frequency matrix
    :param lambda: parameter that determines the amount of contribution of the document vs the collection
    :return: Jelinek Mercer smoothed matrix
    """
    np.seterr(divide='ignore', invalid='ignore')
    return np.log((np.nan_to_num(lamda * tf / tf.sum(axis=0, dtype=float)).T + (1 - lamda) * tf.sum(axis=1) / tf.sum(dtype=float)).T)



def dirichlet_prior_smoothing(tf, mu):
    """
    Assumes p(w|C) = tf(w;C)/|C|

    :param tf:
    :param mu:
    :return:
    """
    return np.log((tf.T + mu * (tf.sum(axis=1)) / tf.sum()).T / (tf.sum(axis=0) + mu))


def absolute_discounting(tf, delta):
    """
    Assumes p(w|C) = tf(w;C)/|C|

    :param tf:
    :param delta:
    :return:
    """
    d_length = tf.sum(axis=0, dtype=float)
    return np.log(np.nan_to_num((tf - delta).clip(0) / d_length) + np.outer(((tf.sum(axis=1)) / tf.sum()),
                                                                            (
                                                                            delta * ((tf > 0).sum(axis=0)) / d_length)))


def score_model(model, query_indices):
    return model[query_indices[0:None], :].sum(axis=0)
