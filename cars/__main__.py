inputs: list[tuple[float]] = [(0.0000, 0.0000), (0.1600, 0.1556), (0.2400, 0.3543), (0.2800, 0.3709),
                              (0.3600, 0.4702), (0.4000, 0.4868), (0.5000, 0.5530), (0.5200, 0.6026),
                              (0.6000, 0.6358), (0.6200, 0.3212), (0.6600, 0.7185), (0.7000, 0.7351),
                              (0.7600, 0.8013), (0.8400, 0.7848), (0.9600, 0.9669), (1.0000, 1.0000)]
targets: list[int] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]  # 0 = keep, 1 = sell

weights: list[float] = [0.1, 0.2]
b: float = 0.3
epochs: int = 400
learning_rate: float = 0.5


def predict(inputs_data):
    return sum([w * i for w, i in zip(weights, inputs_data)]) + b


def activate(x):
    import math
    return 1 / (1 + math.exp(-x))


def log_loss(act, target):
    import math
    return -target * math.log(act) - (1 - target) * math.log(1 - act)


# train the network
for epoch in range(epochs):
    pred = [predict(inp) for inp in inputs]
    act = [activate(p) for p in pred]
    cost = sum([log_loss(a, t) for a, t in zip(act, targets)]) / len(act)
    print(f"ep: {epoch}, c: {cost:.2f};")

    errors_d = [(a - t) for a, t in zip(act, targets)]
    weights_d = [[e * i for i in inp] for e, inp in zip(errors_d, inputs)]
    weights_d_T = list(zip(*weights_d))  # transpose weight_d
    for i in range(len(weights)):
        weights[i] -= learning_rate * sum(weights_d_T[i])/len(weights_d)
    bias_d = [e * 1 for e in errors_d]
    b -= learning_rate * sum(bias_d) / len(bias_d)

test_inputs = [(0.1600, 0.1391), (0.5600, 0.3046),
    (0.7600, 0.8013), (0.9600, 0.3046), (0.1600, 0.7185)]
test_targets = [0, 0, 1, 0, 0] # 0 = keep, 1 = sell

pred = [predict(inp) for inp in test_inputs]
act = [activate(p) for p in pred]

for a, t in zip(act, test_targets):
    print(f"Targets: ${t}, Predicted: ${a:.0f}")
