Contributing to seaborn
=======================

General support
---------------

General support questions ("how do I do X?") are most at home on [StackOverflow](https://stackoverflow.com/), which has a larger audience of people who will see your post and may be able to offer assistance. Your chance of getting a quick answer will be higher if you include runnable code, a precise statement of what you are hoping to achieve, and a clear explanation of the problems that you have encountered.

Reporting bugs
--------------

If you think you've encountered a bug in seaborn, please report it on the [Github issue tracker](https://github.com/mwaskom/seaborn/issues/new). To be useful, bug reports *must* include the following information:

- A reproducible code example that demonstrates the problem
- The output that you are seeing (an image of a plot, or the error message)
- A clear explanation of why you think something is wrong
- The specific versions of seaborn and matplotlib that you are working with

Bug reports are easiest to address if they can be demonstrated using one of the example datasets from the seaborn docs (i.e. with `seaborn.load_dataset`). Otherwise, it is preferable that your example generate synthetic data to reproduce the problem. If you can only demonstrate the issue with your actual dataset, you will need to share it, ideally as a csv (do not share data as a pickle file).

If you've encountered an error, searching the specific text of the message before opening a new issue can often help you solve the problem quickly and avoid making a duplicate report.

Because matplotlib handles the actual rendering, errors or incorrect outputs may be due to a problem in matplotlib rather than one in seaborn. It can save time if you try to reproduce the issue in an example that uses only matplotlib, so that you can report it in the right place. But it is alright to skip this step if it's not obvious how to do it.


New features
------------

If you think there is a new feature that should be added to seaborn, you can open an issue to discuss it. But please be aware that current development efforts are mostly focused on standardizing the API and internals, and there may be relatively low enthusiasm for novel features that do not fit well into short- and medium-term development plans.
