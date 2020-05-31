"""Execute the scripts that comprise the example gallery in the online docs."""
from glob import glob
import matplotlib.pyplot as plt

if __name__ == "__main__":

    fnames = sorted(glob("examples/*.py"))

    for fname in fnames:

        print(f"- {fname}")
        with open(fname) as fid:
            exec(fid.read())
        plt.close("all")
