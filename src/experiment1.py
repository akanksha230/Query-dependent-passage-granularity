import pickle
import numpy as np

G = ["sent", "p128", "p256", "p512", "para", "section"]
BUDGETS = [256, 512, 1024]

cache = pickle.load(open("cache_test.pkl", "rb"))

print("\nEXPERIMENT 1 — FIXED GRANULARITY")
print("Test set: evidence coverage\n")

print(f"{'Granularity':<12} {'B=256':>10} {'B=512':>10} {'B=1024':>10}")
print("-" * 46)

for i, g in enumerate(G):
    scores = []

    for B in BUDGETS:
        score = cache["out"][B]["cov"].mean(0)[i]
        scores.append(score)

    print(
        f"{g:<12} "
        f"{scores[0]:>10.4f} "
        f"{scores[1]:>10.4f} "
        f"{scores[2]:>10.4f}"
    )

print("\nBEST GRANULARITY")

for B in BUDGETS:
    scores = cache["out"][B]["cov"].mean(0)
    best_i = np.argmax(scores)

    print(
        f"B={B}: {G[best_i]} "
        f"(coverage = {scores[best_i]:.4f})"
    )