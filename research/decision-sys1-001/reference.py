"""Independent NumPy reference of the changed feedback operator ONLY.
Not a reference or proof of the inherited observation/prediction graph.
"""
import numpy as np


def correction(state, action_input, outcome, prediction, available, weights):
    f, s, matrix = (np.asarray(x) for x in state)
    mask = np.asarray(available, dtype=bool)
    linear = lambda x, name: x @ weights[name + '.weight'].T + weights[name + '.bias']
    sigmoid = lambda x: .5 * (1 + np.tanh(.5 * x))
    address = lambda x: x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + 1e-5)
    action_input = np.where(mask, action_input, 0)
    residual = np.tanh(np.where(mask, outcome, 0) - np.where(mask, prediction, 0))
    surprise = np.mean(residual * residual, axis=-1, keepdims=True)
    action = np.tanh(linear(action_input, 'action'))
    joint = np.concatenate((f, s), axis=-1)
    proposal, gate = np.split(linear(np.concatenate((joint, action, residual), axis=-1), 'verify_map'), 2, axis=-1)
    delta = .25 * surprise * sigmoid(gate) * np.tanh(proposal)
    active = mask & (surprise > 0)
    corrected = np.where(active, joint + delta, joint)
    key = address(linear(action, 'verify_key'))
    value = np.tanh(linear(residual, 'verify_value'))
    error = value - np.einsum('bij,bj->bi', matrix, key)
    update = .25 * surprise[..., None] * error[..., None] * key[:, None, :]
    corrected_matrix = np.where(active[..., None], matrix + update, matrix)
    return corrected[:, :32], corrected[:, 32:], corrected_matrix
