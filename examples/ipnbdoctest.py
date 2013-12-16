#!/usr/bin/env python
"""
simple example script for running and testing notebooks.

Usage: `ipnbdoctest.py foo.ipynb [bar.ipynb [...]]`

Each cell is submitted to the kernel, and the outputs are compared with those
stored in the notebook.

From https://gist.github.com/minrk/2620735

"""
from __future__ import print_function
import os
import sys
import re
import difflib

from collections import defaultdict
from Queue import Empty
from StringIO import StringIO

from IPython.kernel import KernelManager
from IPython.nbformat.current import reads, NotebookNode

SKIP_COMPARE = ('traceback', 'latex', 'prompt_number')
IMAGE_OUTPUTS = ('png', 'svg', 'jpeg')


def sanitize(s):
    """Sanitize a string for comparison.

    - Fix universal newlines
    - Strip trailing newlines
    - Normalize likely random values (memory addresses and UUIDs)

    """
    if not isinstance(s, basestring):
        return s

    # Formalize newline:
    s = s.replace('\r\n', '\n')

    # Ignore trailing newlines (but not space)
    s = s.rstrip('\n')

    # Normalize hex addresses:
    s = re.sub(r'0x[a-f0-9]+', '0xFFFFFFFF', s)

    # Normalize UUIDs:
    s = re.sub(r'[a-f0-9]{8}(\-[a-f0-9]{4}){3}\-[a-f0-9]{12}', 'U-U-I-D', s)

    return s


def consolidate_outputs(outputs):
    """consolidate outputs into a summary dict (incomplete)"""
    data = defaultdict(list)
    data['stdout'] = ''
    data['stderr'] = ''

    for out in outputs:
        if out.type == 'stream':
            data[out.stream] += out.text
        elif out.type == 'pyerr':
            data['pyerr'] = dict(ename=out.ename, evalue=out.evalue)
        else:
            for key in ('png', 'svg', 'latex', 'html',
                        'javascript', 'text', 'jpeg',):
                if key in out:
                    data[key].append(out[key])
    return data


def base64_to_array(data):
    """Convert a base64 image to an array."""
    import numpy as np
    import Image
    return np.array(Image.open(StringIO(data.decode("base64")))) / 255.


def image_diff(test, ref, key="image", prompt_num=None):
    """Diff two base64-encoded images."""
    if test == ref:
        return True, ""

    message = "Mismatch in %s output" % key
    if prompt_num is not None:
        message += " (#%d)" % prompt_num

    try:
        test = base64_to_array(test)
        ref = base64_to_array(ref)
        if test.shape == ref.shape:
            import numpy as np
            diff = np.abs(test - ref).sum()
            diff /= len(diff.flat) * 100
            # TODO hardcode eps, make configurable later
            if diff < 1:
                return True, ""
            message += ": %.3g%% difference" % diff
        else:
            message += ": Test image (%dx%d); " % test.shape[:2]
            message += "; Ref image (%dx%d)" % ref.shape[:2]
    except ImportError:
        pass
    return False, message


def compare_outputs(test, ref, prompt_num=None, skip_compare=SKIP_COMPARE):
    """Test whether the stored outputs match the execution outputs."""
    match, message = True, ""

    # Iterate through the reference output fields
    for key in ref:

        # Don't check everything
        if key in skip_compare:
            continue

        # Report when test output is missing a field
        if key not in test:
            match = False
            msg = "Mismatch: '%s' field not in test output" % key
            if prompt_num is not None:
                msg += " (#%d)" % prompt_num
            message += msg + "\n"
            continue

        # Obtain the field values
        test_value = test[key]
        ref_value = ref[key]

        # Diff images seperately
        if key in IMAGE_OUTPUTS:

            mtch, msg = image_diff(test_value, ref_value, key, prompt_num)
            match = match and mtch
            message += msg

        else:

            # Clean up some randomness and check the match
            test_value = sanitize(test_value)
            ref_value = sanitize(ref_value)
            if test_value == ref_value:
                continue

            # Build a textual diff report
            match = False
            diff = difflib.context_diff(test_value.split("\n"),
                                        ref_value.split("\n"),
                                        "Test output",
                                        "Reference output",
                                        n=1, lineterm="")
            message += "Mismatch in textual output"
            if prompt_num is not None:
                message += " (#%d)\n" % prompt_num
            message += "\n  ".join(diff) + "\n"

    return match, message


def run_cell(shell, iopub, cell):
    # print cell.input
    shell.execute(cell.input)
    # wait for finish, maximum 20s
    shell.get_msg(timeout=20)
    outs = []

    while True:
        try:
            msg = iopub.get_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue

        content = msg['content']
        # print msg_type, content
        out = NotebookNode(output_type=msg_type)

        if msg_type == 'stream':
            out.stream = content['name']
            out.text = content['data']
        elif msg_type in ('display_data', 'pyout'):
            out['metadata'] = content['metadata']
            for mime, data in content['data'].iteritems():
                attr = mime.split('/')[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(out, attr, data)
            if msg_type == 'pyout':
                out.prompt_number = content['execution_count']
        elif msg_type == 'pyerr':
            out.ename = content['ename']
            out.evalue = content['evalue']
            out.traceback = content['traceback']
        else:
            print("unhandled iopub msg:", msg_type)

        outs.append(out)
    return outs


def test_notebook(nb):
    """Main function to run tests at the level of one notebook."""
    # Boot up the kernel, assume inline plotting
    km = KernelManager()
    km.start_kernel(extra_arguments=["--matplotlib=inline",
                                     "--colors=NoColor"],
                    stderr=open(os.devnull, 'w'))

    # Connect, allowing for older IPythons
    try:
        kc = km.client()
        kc.start_channels()
        iopub = kc.iopub_channel
    except AttributeError:
        # IPython 0.13
        kc = km
        kc.start_channels()
        iopub = kc.sub_channel
    shell = kc.shell_channel

    # Initialize the result tracking
    successes = 0
    failures = 0
    errors = 0
    fail_messages = []
    err_messages = []

    # Iterate the notebook, testing only code cells
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue

            # Try and get the prompt number for easier reference
            try:
                prompt_num = cell.prompt_number
            except AttributeError:
                prompt_num = None

            # Try to execute the cell, catch errors from test execution
            try:
                outs = run_cell(shell, iopub, cell)
            except Exception as e:
                message = "Error while running cell:\n%s" % repr(e)
                err_messages.append(message)
                errors += 1
                sys.stdout.write("E")
                continue

            errored = False
            failed = False

            for out, ref in zip(outs, cell.outputs):

                # Now check for an error in the cell execution itself
                bad_error = (out.output_type == "pyerr"
                             and not ref.output_type == "pyerr")
                if bad_error:
                    message = "\nError in code cell"
                    if prompt_num is not None:
                        message = " %s (#%d)" % (message, prompt_num)
                    message = "%s:\n%s" % (message, "".join(out.traceback))
                    err_messages.append(message)
                    errored = True

                # Otherwise check whether the stored and achived outputs match
                else:
                    try:
                        match, message = compare_outputs(out, ref, prompt_num)
                        if not match:
                            failed = True
                            fail_messages.append(message)

                    except Exception as e:
                        message = "Error while comparing output:\n%s" % repr(e)
                        err_messages.append(message)
                        errors += 1
                        sys.stdout.write("E")
                        continue

            if failed:
                failures += 1
                dot = "F"
            elif errored:
                errors += 1
                dot = "E"
            else:
                successes += 1
                dot = "."
            sys.stdout.write(dot)

    print()
    print("    %3i cells successfully replicated" % successes)
    if failures:
        print("    %3i cells mismatched output" % failures)
        print("\n" + "\n".join(fail_messages) + "\n")
    if errors:
        print("    %3i cells failed to complete" % errors)
        print("\n" + "\n".join(err_messages) + "\n")
    kc.stop_channels()
    km.shutdown_kernel()
    del km

    return int(bool(failures + errors))

if __name__ == '__main__':

    status = 0
    for ipynb in sys.argv[1:]:
        print("testing %s" % ipynb)
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')

        status += test_notebook(nb)
    sys.exit(status)
