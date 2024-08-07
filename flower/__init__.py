import data

weights: list[list[float]] = [[0.1, 0.2], [0.15, 0.25], [0.18, 0.1]]
biases: list[float] = [0.3, 0.4, 0.35]
epochs: int = 5000
learning_rate: float = .5   


def softmax(predictions) -> list[float]:
    import math
    m: float = max(predictions)
    temp: list[float] = [math.exp(p - m) for p in predictions]
    total: float = sum(temp)
    return [t / total for t in temp]


def log_loss(activations, targets):
    import math
    losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
    return sum(losses)


# training the network
for epoch in range(epochs):
    pred: list[list[float]] = [[sum([w * i for w, i in zip(we, inp)]) +
             bi for we, bi in zip(weights, biases)] for inp in data.inputs]
    act: list[list[float]] = [softmax(p) for p in pred]
    cost: float = sum([log_loss(ac, ta) for ac, ta in zip(act, data.targets)]) / len(act)
    errors_d: list[list[float]] = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act, data.targets)]
    inputs_T: list[list[float]] = list(zip(*data.inputs))
    errors_d_T: list[list[float]] = list(zip(*errors_d))
    weights_d: list[list[float]] = [[sum([e * i for e, i in zip(er, inp)]) for er in errors_d_T] for inp in inputs_T]
    biases_d: list[int] = [sum([e for e in errors]) for errors in errors_d_T]
    weights_d_T: list[list[float]] = list(zip(*weights_d))
    for y in range(len(weights_d_T)):
        for x in range(len(weights_d_T[0])):
            weights[y][x] -= learning_rate * weights_d_T[y][x] / len(data.inputs)
        biases[y] -= learning_rate * biases_d[y] / len(data.inputs)

# test the network
pred: list[list[float]] = [[sum([w * i for w, i in zip(we, inp)]) +
         bi for we, bi in zip(weights, biases)] for inp in data.test_inputs]

act: list[list[float]] = [softmax(p) for p in pred]
correct: int = 0
for a, t in zip(act, data.test_targets):
    if a.index(max(a)) == t.index(max(t)):
        correct += 1
print(f"Correct: {correct}/{len(act)} ({correct/len(act):%})")
