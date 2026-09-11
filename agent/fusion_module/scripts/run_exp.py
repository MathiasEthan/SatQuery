# scripts/run_experiments.py
import pandas as pd
from src.train import train_one, CONFIGS


def main():
    rows = []

    for name in CONFIGS:
        for seed in [0, 1, 2]:
            print(f"\n=== {name} seed={seed} ===")
            rows.append(train_one(name, seed=seed, verbose=True))

    for seed in [0, 1, 2]:
        print(f"\n=== fused SHUFFLED-radar control seed={seed} ===")
        rows.append(train_one("fused", seed=seed, verbose=True, shuffle_s1=True))

    df = pd.DataFrame(rows)
    df["run"] = df["config"] + df["shuffled"].map({True: "_shuffled", False: ""})

    print("\n\n=== per run ===")
    print(df.to_string(index=False))

    print("\n=== mean +/- std over seeds ===")
    print(df.groupby("run")[["macro_f1", "micro_f1", "mAP"]]
            .agg(["mean", "std"]).round(4).to_string())

    df.to_csv("results_stage1.csv", index=False)


if __name__ == "__main__":
    main()