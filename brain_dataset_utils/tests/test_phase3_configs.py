"""Guard: the Phase 3 arm configs are internally consistent.

A config that asks for structure weighting but loads a dataset with no labels
builds the model, fills RAM with 24k slices, and only then dies one minute into
training -- after the GPU allocation is already spent. It happened: the config
generator used the wrong indentation for `dataset_type`, str.replace() matched
nothing, and three configs were written with the label-less dataset.
"""
import glob
import os
import re
import sys

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')

EXPECT_START = 'top_model_epoch_472.pth'
EXPECT_EPOCHS = 532


def parse(path):
    """Flat scalar lookup -- the configs use !!python/tuple, so no yaml.safe_load."""
    out = {}
    for line in open(path, encoding='utf-8'):
        m = re.match(r"\s*([A-Za-z_]+):\s*'?([^'#\n]*?)'?\s*(?:#.*)?$", line)
        if m and m.group(2).strip():
            out.setdefault(m.group(1), m.group(2).strip())
    return out


def main():
    paths = sorted(glob.glob(os.path.join(REPO, 'configs', 'phase3_*.yaml')))
    if not paths:
        print("no phase3 configs found"); sys.exit(1)

    fails = 0
    names = set()
    for p in paths:
        rel = os.path.relpath(p, REPO)
        c = parse(p)
        dg = float(c.get('lambda_deepgray_weight', 1.0))
        dtype = c.get('dataset_type', '')

        if dg > 1.0 and not dtype.endswith('_labels'):
            print(f"  FAIL {rel}: lambda_deepgray_weight={dg} but dataset_type="
                  f"{dtype!r} carries no labels"); fails += 1
        if dg > 1.0 and c.get('deepgray_mode') not in ('uniform', 'shell'):
            print(f"  FAIL {rel}: deepgray_mode={c.get('deepgray_mode')!r}"); fails += 1
        if int(c.get('n_epochs', 0)) != EXPECT_EPOCHS:
            print(f"  FAIL {rel}: n_epochs={c.get('n_epochs')} (absolute; expected "
                  f"{EXPECT_EPOCHS} = 472 + 60)"); fails += 1
        if EXPECT_START not in c.get('model_load_path', ''):
            print(f"  FAIL {rel}: does not start from {EXPECT_START}"); fails += 1
        if 'fine_v2' not in c.get('dataset_path', ''):
            print(f"  FAIL {rel}: dataset_path={c.get('dataset_path')!r} is not fine_v2"
                  f" (the only build carrying LABEL_dataset)"); fails += 1
        names.add(c.get('dataset_path', ''))
        print(f"  checked {rel}: deepgray={dg}, mode={c.get('deepgray_mode')}, "
              f"dataset={dtype.split('_')[-1]}")

    # All arms must share a data path, or the comparison is not controlled.
    if len(names) > 1:
        print(f"  FAIL arms disagree on dataset_path: {names}"); fails += 1

    if fails:
        print(f"{fails} config problem(s)"); sys.exit(1)
    print("phase 3 arm configs are consistent")


if __name__ == '__main__':
    main()
