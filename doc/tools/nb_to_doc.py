#! /usr/bin/env python
"""Execute a .ipynb file, write out a processed .rst and clean .ipynb.

The functions in this script were copied from the nbstripout tool:

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


def _cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell


def determine_keep_output(cell, default):
    """Given a cell, determine whether output should be kept
    Based on whether the metadata has "init_cell": true,
    "keep_output": true, or the tags contain "keep_output" """
    if 'init_cell' in cell.metadata:
        return bool(cell.metadata.init_cell)

    has_keep_output_metadata = 'keep_output' in cell.metadata
    keep_output_metadata = bool(cell.metadata.get('keep_output', False))

    has_keep_output_tag = 'keep_output' in cell.metadata.get('tags', [])

    # keep_output between metadata and tags should not contradict each other
    if has_keep_output_metadata \
       and has_keep_output_tag \
       and not keep_output_metadata:
        raise MetadataError(
            "cell metadata contradicts tags: "
            "\"keep_output\": false, but keep_output in tags"
        )

    if has_keep_output_metadata or has_keep_output_tag:
        return keep_output_metadata or has_keep_output_tag
    return default


def strip_output(nb, keep_output=False, keep_count=False, extra_keys=''):
    """
    Strip the outputs, execution count/prompt number and miscellaneous
    metadata from a notebook object, unless specified to keep either the
    outputs or counts.
    `extra_keys` could be 'metadata.foo cell.metadata.bar metadata.baz'
    """
    if keep_output is None and 'keep_output' in nb.metadata:
        keep_output = bool(nb.metadata['keep_output'])

    if hasattr(extra_keys, 'decode'):
        extra_keys = extra_keys.decode()
    extra_keys = extra_keys.split()
    keys = {'metadata': [], 'cell': {'metadata': []}}
    for key in extra_keys:
        if key.startswith('metadata.'):
            keys['metadata'].append(key[len('metadata.'):])
        elif key.startswith('cell.metadata.'):
            keys['cell']['metadata'].append(key[len('cell.metadata.'):])
        else:
            sys.stderr.write('ignoring extra key `%s`' % key)

    nb.metadata.pop('signature', None)
    nb.metadata.pop('widgets', None)

    for field in keys['metadata']:
        pop_recursive(nb.metadata, field)

    for cell in _cells(nb):
        keep_output_this_cell = determine_keep_output(cell, keep_output)

        # Remove the outputs, unless directed otherwise
        if 'outputs' in cell:

            # Default behavior strips outputs. With all outputs stripped,
            # there are no counts to keep and keep_count is ignored.
            if not keep_output_this_cell:
                cell['outputs'] = []

            # If keep_output_this_cell, but not keep_count, strip the counts
            # from the output.
            if keep_output_this_cell and not keep_count:
                for output in cell['outputs']:
                    if 'execution_count' in output:
                        output['execution_count'] = None

            # If keep_output_this_cell and keep_count, do nothing.

        # Remove the prompt_number/execution_count, unless directed otherwise
        if 'prompt_number' in cell and not keep_count:
            cell['prompt_number'] = None
        if 'execution_count' in cell and not keep_count:
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
    _, fpath = sys.argv
    basedir, fname = os.path.split(fpath)
    fstem = fname[:-6]

    # Read the notebook
    print(f"Executing {fpath} ...", end=" ", flush=True)
    with open(fpath) as f:
        nb = nbformat.read(f, as_version=4)

    # Run the notebook
    kernel = os.environ.get("NB_KERNEL", None)
    if kernel is None:
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
    print(f"Writing clean {fpath} ... ", end=" ", flush=True)
    nb = strip_output(nb)
    with open(fpath, "wt") as f:
        nbformat.write(nb, f)

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
