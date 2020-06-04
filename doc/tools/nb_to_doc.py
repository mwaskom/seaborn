#! /usr/bin/env python
import os
import sys
import nbformat
from nbconvert import RSTExporter
from nbconvert.preprocessors import (
    ExecutePreprocessor,
    TagRemovePreprocessor,
    ExtractOutputPreprocessor
)
from traitlets.config import Config


if __name__ == "__main__":

    # Get the desired ipynb file path and parse into components
    _, fpath = sys.argv
    basedir, fname = os.path.split(fpath)
    fstem = fname[:-6]

    # Read the notebook
    print(f"Executing {fpath} ...", end=" ", flush=True)
    with open(fpath) as f:
        nb = nbformat.read(f, as_version=4)

    # Run the notebook
    kernel = nb["metadata"]["kernelspec"]["name"]
    ep = ExecutePreprocessor(
        timeout=600,
        kernel_name=kernel,
        extra_arguments=["--InlineBackend.rc={'figure.dpi': 96}"]
    )
    ep.preprocess(nb, {"metadata": {"path": basedir}})

    # Remove the execution result outputs
    for cell in nb.get("cells", {}):
        fields = cell.get("outputs", [])
        for field in fields:
            if field["output_type"] == "execute_result":
                fields.remove(field)

    # TODO write a clean notebook file back out?

    # Convert to .rst formats
    exp = RSTExporter()

    c = Config()
    c.TagRemovePreprocessor.remove_cell_tags = {"hide"}
    c.TagRemovePreprocessor.remove_input_tags = {"hide-input"}
    c.TagRemovePreprocessor.remove_all_outputs_tags = {"hide-output"}
    c.ExtractOutputPreprocessor.output_filename_template = \
        f"{fstem}_files/{fstem}_" + "{cell_index}_{index}{extension}"

    exp.register_preprocessor(TagRemovePreprocessor(config=c), True)
    exp.register_preprocessor(ExtractOutputPreprocessor(config=c), True)

    body, resources = exp.from_notebook_node(nb)

    # Write the .rst file
    rst_path = os.path.join(basedir, f"{fstem}.rst")
    print(f"Writing {rst_path}")
    with open(rst_path, "w") as f:
        f.write(body)

    # Write the individual image outputs
    imdir = os.path.join(basedir, f"{fstem}_files")
    if not os.path.exists(imdir):
        os.mkdir(imdir)

    for imname, imdata in resources["outputs"].items():
        if imname.startswith(fstem):
            impath = os.path.join(basedir, f"{imname}")
            with open(impath, "wb") as f:
                f.write(imdata)
