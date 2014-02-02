Contributing code
=================

To contribute code to seaborn, it's best to follow the usual github workflow:

- Fork the [https://github.com/mwaskom/seaborn](main seaborn repository)
- Create a feature branch with `git checkout -b <feature_name>`
- Add some new code
- Push to your fork with `git push origin <feature_name>`
- Open a pull-request on the main repository

Here are some further notes on specific aspects of seaborn development that are good to know about.

#### Getting in touch

In general, it can't hurt to get in touch by opening an issue before you start your work. Because seaborn is relatively young, there are a lot of things that I have partially-formed thoughts on, but haven't gotten a chance to fully implement yet. I very much appreciate help, but I'll be more likely to merge in changes that fit into my plans for the package (which might only exist inside my head). So, giving me a heads up about what you have in mind will save time for everyone.

#### Where to branch

For any new features, or enhancements to existing features, you should branch off `master`. The main repo also has branches corresponding to each point release (e.g. `v0.2`). If you are fixing a bug, it might be better to branch from there so the fix can be included in an incremental release. This will probably get sorted out in the issue reporting the bug.

#### Working on a Pull Request

Since seaborn is a plotting package, it's most useful to be able to see the new feature or the consequences of changes your contribution will make. When you open the pull request, including a link to an example notebook (through [nbviewer](http://nbviewer.ipython.org/)) or at least a static screenshot is very helpful.

#### Testing and documentation

Currently, seaborn uses the notebooks in `examples/` for both documentation and testing. This is proving to be a somewhat problematic solution, and I am worried about many incremental changes to these notebooks producing a large and unwieldy repository. Please try to hold off committing changes to the notebooks until the feature is ready to go. In the meantime, it might be useful to discuss changes in the context of the example notebooks, but please edit them without committing and share via nbviewer from a gist/dropbox link/etc.

The formal unit-test coverage of the package is quite poor, as the focus has been on using the example notebooks for testing. Going forward, this should change. Please include unit-tests that at least touch the various branches through the functions to ward off errors; in cases where it's possible to programmatically check the outputs of the functions, please do so.

Once you're ready to update the docs, it's good to add a little narrative information about what a feature does and what kind of visualization problems it can be useful for. Then, provide an example or two showing the function in action. The existing docs should be a good guide here.

If you're unsure where in the documentation your feature should be discussed, please feel free to ask.

After adding your changes but before committing, please perform the following to steps:

- Restart the notebook kernel and "run all" cells so you can be certain the notebook executes and the cell numbers are in the right order

- Run `make hexstrip` to remove the random hex memory identifiers that are stored in the notebook, for a cleaner commit

- Use `git diff` to make sure your changes didn't result in a cascading change to lots of figures

Useful commands to know about for testing:

- `make test` runs the full test suite (unit-tests and notebooks)

- `nosetests` runs the unit-test suite in isolation

- `python examples/ipnbdoctest.py examples/<notebook>.ipynb` can be used to test a specific notebook

- `make coverage` will run the unit-test suite and produce a coverage report

- `make lint` will run `pep8` and `pyflakes` over the codebase. Doing so requires [this](https://github.com/dcramer/pyflakes) fork of pyflakes, which can be installed with `pip install https://github.com/dcramer/pyflakes/tarball/master`

Functions should be documented with the [numpy](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt) standard. Current functions usually don't have examples, but it would be more useful if they did.
