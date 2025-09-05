
import subprocess
import optuna
import os
import json
import tempfile
from datetime import datetime
import csv

CONFIG_PATH = "./config/experiment_config.json"
TRIAL_LOG_PATH = "./results/optuna_trial_log.csv"
#Choose the appropriate path

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def write_trial_log(trial_number, strengths, asr, layer_ids, ngr=None, score=None):
    file_exists = os.path.isfile(TRIAL_LOG_PATH)
    with open(TRIAL_LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        headers = ["Trial"] + [f"alpha_{l}" for l in layer_ids] + ["ASR"]
        row = [trial_number] + strengths + [asr]
        if ngr is not None:
            headers += ["NGR", "Score"]
            row += [ngr, score]
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)


def run_intervention_script(config, strengths, layers, save_path):
    strength_args = [str(s) for s in strengths]
    layer_args = [str(l) for l in layers]

    command = [
        "python", config["intervention_script"],
        "--datasets", config["datasets"],
       # "--text_columns", config["text_columns"],
        "--anchors", *config["anchors"],
        "--separation_file", save_path,
        "--intervention_layers", *layer_args,
        "--intervention_strengths", *strength_args,
        "--precomputed_directions", config["diff_vector_cache"],
        "--seed", str(config.get("seed", 42))
    ]
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("[Debugging] Intervene in script output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("[Error] Intervention script execution failed:")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        raise


def evaluate_with_llamaguard(config, json_file):
    command = [
        "python", config["evaluator_script"],
        "--json_file", json_file
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if "ASR" in line:
            return float(line.split()[1])
    raise ValueError("ASR not found in output")


def evaluate_ngram_repetition(json_file):
    command = [
        "python", "compute_ngram_repetition.py",
        "--json_file", json_file
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in result.stdout.splitlines():
        if "n-gram repeat rate" in line:
            return float(line.split()[-1])
    raise ValueError("Repeat rate not found in output")


def precompute_directions_if_needed(config):
    if os.path.exists(config["diff_vector_cache"]):
        print("[Cache] Difference vector already exists, no need to recalculate")
        return

    print("[Preprocessing] Calculating difference vector...")
    tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
    command = [
        "python", config["intervention_script"],
        "--datasets", config["datasets"],
      #  "--text_columns", config["text_columns"],
        "--anchors", *config["anchors"],
        "--separation_file", tmp_output,
        "--intervention_layers", *[str(l) for l in config["intervention_layers"]],
        "--intervention_strengths", *["0.0"] * len(config["intervention_layers"]),
        "--save_directions", config["diff_vector_cache"],
        "--seed", str(config.get("seed", 42))
    ]
    subprocess.run(command, check=True)
    os.remove(tmp_output)


def main():
    config = load_config()
    output_dir = config.get("output_dir", "./results/by_trial")
    os.makedirs(output_dir, exist_ok=True)
    intervention_layers = config["intervention_layers"]

    def objective(trial):
        strengths = [trial.suggest_float(f"alpha_{l}", 0.0, 1.0) for l in intervention_layers]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"trial_{trial.number}_{timestamp}.json")
        run_intervention_script(config, strengths, intervention_layers, output_path)

        try:
            asr = evaluate_with_llamaguard(config, output_path)
            ngr = evaluate_ngram_repetition(output_path)
            score = asr - ngr
        except Exception as e:
            print(f"error: {e}")
            return 0.0
        print(f"Trial {trial.number}: strengths = {strengths}, ASR = {asr:.4f}, NGR = {ngr:.4f}, Score = {score:.4f}")
        write_trial_log(trial.number, strengths, asr, intervention_layers, ngr=ngr, score=score)
        return score

    precompute_directions_if_needed(config)
   # sampler = optuna.samplers.TPESampler(seed=42)
   # study = optuna.create_study(direction="maximize", sampler=sampler)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials=config.get("n_trials", 30),
        callbacks=[optuna.study.MaxTrialsCallback(config.get("n_trials", 30))],
        show_progress_bar=True
    )

    print("Best trial:")
    print(f"  Value (Score): {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")


if __name__ == "__main__":
    main()
