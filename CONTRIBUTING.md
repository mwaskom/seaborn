Contributing to seaborn
=======================

General support
---------------

General support questions are most at home on [StackOverflow](http://stackoverflow.com/), where they will be seen by more people and are more easily searchable. StackOverflow has a `seaborn` tag, which will bring the question to the attention of people who might be able to answer.

Reporting bugs
--------------

If you think you have encountered a bug in seaborn, please report it on the [Github issue tracker](https://github.com/mwaskom/seaborn/issues/new). It will be most helpful to include a reproducible script with one of the example datasets (accessed through `load_dataset()`) or using some randomly-generated data.

It is difficult debug any issues without knowing the versions of seaborn and matplotlib you are using, as well as what matplotlib backend you are using to draw the plots, so please include those in your bug report.

Fixing bugs
-----------

If you know how to fix a bug you have encountered or see on the issue tracker, that is very appreciated. Please submit a [pull request](https://help.github.com/articles/using-pull-requests/) on the main seaborn repository with the fix. The presence of a bug implies a lack of coverage in the tests, so when fixing a bug, it is best to add a test that fails before the fix and passes after to make sure it does not reappear. See the section on testing below. But if there is an obvious fix and you're not sure how to write a test, don't let that stop you.

Documentation issues
--------------------

If you see something wrong or confusing in the documentation, please report it with an issue or fix it and open a pull request.

New features
------------

If you'd like to add a new feature to seaborn, it's best to open an issue to discuss it first. Given the nature of seaborn's goals and approach, it can be hard to write a substantial contribution that is consistent with the rest of the package, and I often lack the bandwidth to help. Also, every new feature represents a new commitment for support. For these reasons, I'm somewhat averse to large feature contributions. Smaller or well-targeted enhancements can be helpful and should be submitted through the normal pull-request workflow. Please include tests for any new features and make sure your changes don't break any existing tests.

Testing seaborn
---------------

Seaborn is primarily tested through a `nose` unit-test suite that interacts with the private objects that actually draw the plots behind the function interface. The basic approach here is to test the numeric information going into and coming out of the matplotlib functions. Currently, there is a general assumption that matplotlib is drawing things properly, and tests are run against the data that ends up in the matplotlib objects but not against the images themselves. See the existing tests for examples of how this works.

To execute the test suite and doctests, run `make test` in the root source directory. You can also build a test coverage report with `make coverage`. 

The `make lint` command will run `pep8` and `pyflakes` over the codebase to check for style issues. Doing so requires [this](https://github.com/dcramer/pyflakes) fork of pyflakes, which can be installed with `pip install https://github.com/dcramer/pyflakes/tarball/master`. It also currently requires `pep8` 1.5 or older, as the rules got stricter and the codebase has not been updated. This is part of the Travis build, and the build will fail if there are issues, so please do this before submitting a pull request.
