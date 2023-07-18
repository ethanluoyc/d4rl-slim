import d4rl.infos
import json


def main():
    infos = {}
    for dataset_name in sorted(d4rl.infos.DATASET_URLS.keys()):
        infos[dataset_name] = {
            "name": dataset_name,
            "dataset_url": d4rl.infos.DATASET_URLS.get(dataset_name, None),
            "ref_min_score": d4rl.infos.REF_MIN_SCORE.get(dataset_name, None),
            "ref_max_score": d4rl.infos.REF_MAX_SCORE.get(dataset_name, None),
        }

    with open("d4rl_infos.json", "w") as f:
        json.dump(infos, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
