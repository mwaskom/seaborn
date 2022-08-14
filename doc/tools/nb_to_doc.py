#! /usr/bin/env python
"""Execute a .ipynb file, write out a processed .rst and clean .ipynb.

Some functions in this script were copied from the nbstripout tool:

Copyright (c) 2015 Min RK, Florian Rathgeber, Michael McNeil Forbes
2019 Casper da Costa-Luis

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
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


class MetadataError(Exception):
    pass


def pop_recursive(d, key, default=None):
    """dict.pop(key) where `key` is a `.`-delimited list of nested keys.
    >>> d = {'a': {'b': 1, 'c': 2}}
    >>> pop_recursive(d, 'a.c')
    2
    >>> d
    {'a': {'b': 1}}
    """
    nested = key.split('.')
    current = d
    for k in nested[:-1]:
        if hasattr(current, 'get'):
            current = current.get(k, {})
        else:
            return default
    if not hasattr(current, 'pop'):
        return default
    return current.pop(nested[-1], default)


def strip_output(nb):
    """
    Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the
    outputs or counts.
    """
    keys = {'metadata': [], 'cell': {'metadata': ["execution"]}}

    nb.metadata.pop('signature', None)
    nb.metadata.pop('widgets', None)

    for field in keys['metadata']:
        pop_recursive(nb.metadata, field)

    if 'NB_KERNEL' in os.environ:
        nb.metadata['kernelspec']['name'] = os.environ['NB_KERNEL']
        nb.metadata['kernelspec']['display_name'] = os.environ['NB_KERNEL']

    for cell in nb.cells:

        if 'outputs' in cell:
            cell['outputs'] = []
        if 'prompt_number' in cell:
            cell['prompt_number'] = None
        if 'execution_count' in cell:
            cell['execution_count'] = None

        # Always remove this metadata
        for output_style in ['collapsed', 'scrolled']:
            if output_style in cell.metadata:
                cell.metadata[output_style] = False
        if 'metadata' in cell:
            for field in ['collapsed', 'scrolled', 'ExecuteTime']:
                cell.metadata.pop(field, None)
        for (extra, fields) in keys['cell'].items():
            if extra in cell:
                for field in fields:
                    pop_recursive(getattr(cell, extra), field)
    return nb


if __name__ == "__main__":

    # Get the desired ipynb file path and parse into components
    _, fpath, outdir = sys.argv
    basedir, fname = os.path.split(fpath)
    fstem = fname[:-6]

    # Read the notebook
    with open(fpath) as f:
        nb = nbformat.read(f, as_version=4)

    # Run the notebook
    kernel = os.environ.get("NB_KERNEL", None)
    if kernel is None:
        kernel = nb["metadata"]["kernelspec"]["name"]
    ep = ExecutePreprocessor(
        timeout=600,
        kernel_name=kernel,
        extra_arguments=["--InlineBackend.rc=figure.dpi=88"]
    )
    ep.preprocess(nb, {"metadata": {"path": basedir}})

    # Remove plain text execution result outputs
    for cell in nb.get("cells", {}):
        if "show-output" in cell["metadata"].get("tags", []):
            continue
        fields = cell.get("outputs", [])
        for field in fields:
            if field["output_type"] == "execute_result":
                data_keys = field["data"].keys()
                for key in list(data_keys):
                    if key == "text/plain":
                        field["data"].pop(key)
                if not field["data"]:
                    fields.remove(field)

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

    # Clean the output on the notebook and save a .ipynb back to disk
    nb = strip_output(nb)
    with open(fpath, "wt") as f:
        nbformat.write(nb, f)

    # Write the .rst file
    rst_path = os.path.join(outdir, f"{fstem}.rst")
    with open(rst_path, "w") as f:
        f.write(body)

    # Write the individual image outputs
    imdir = os.path.join(outdir, f"{fstem}_files")
    if not os.path.exists(imdir):
        os.mkdir(imdir)

    for imname, imdata in resources["outputs"].items():
        if imname.startswith(fstem):
            impath = os.path.join(outdir, f"{imname}")
            with open(impath, "wb") as f:
                f.write(imdata)
