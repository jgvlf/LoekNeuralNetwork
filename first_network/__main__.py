inputs: list[int] = [1, 2, 3, 4]
targets: list[int] = [2, 4, 6, 8]

w: float = 0.1
learning_rate: float = 0.1
epochs: int = 20


def predict(i):
    return w * i


for _ in range(epochs):
    pred: list[int] = [predict(i) for i in inputs]
    errors: list[int] = [(p - t) ** 2 for p, t in zip(pred, targets)]

    cost: float = sum(errors) / len(targets)
    print(f"Weight: {w:.2f}, Cost: {cost:.2f};")
    errors_d: list[int] = [2 * (p - t) for p, t in zip(pred, targets)]
    weight_d = [e * i for e, i in zip(errors_d, inputs)]
    w -= learning_rate * sum(weight_d) / len(weight_d)

test_inputs: list[int] = [5, 6]
test_targets: list[int] = [10, 12]
pred: list[int] = [predict(i) for i in test_inputs]
for i, t, p in zip(test_inputs, test_targets, pred):
    print(f"Input:{i}, Target:{t}, Pred:{p:.4f};")
