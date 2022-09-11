"""
Cache test datasets before running tests / building docs.

Avoids race conditions that would arise from parallelization.
"""
import pathlib
import re

from seaborn import load_dataset

path = pathlib.Path(".")
py_files = path.rglob("*.py")
ipynb_files = path.rglob("*.ipynb")

datasets = []

for fname in py_files:
    with open(fname) as fid:
        datasets += re.findall(r"load_dataset\(['\"](\w+)['\"]", fid.read())

for p in ipynb_files:
    with p.open() as fid:
        datasets += re.findall(r"load_dataset\(\\['\"](\w+)\\['\"]", fid.read())

for name in sorted(set(datasets)):
    print(f"Caching {name}")
    load_dataset(name)
