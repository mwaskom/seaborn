from inspect import signature, Parameter
from functools import wraps
import warnings


# This function was adapted from scikit-learn
# github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py
def _deprecate_positional_args(f):
    """Decorator for methods that issues warnings for positional arguments.

    Using the keyword-only argument syntax in pep 3102, arguments after the
    * will issue a warning when passed as a positional argument.

    Parameters
    ----------
    f : function
        function to check arguments on

    """
    sig = signature(f)
    kwonly_args = []
    all_args = []

    for name, param in sig.parameters.items():
        if param.kind == Parameter.POSITIONAL_OR_KEYWORD:
            all_args.append(name)
        elif param.kind == Parameter.KEYWORD_ONLY:
            kwonly_args.append(name)

    @wraps(f)
    def inner_f(*args, **kwargs):
        extra_args = len(args) - len(all_args)
        if extra_args > 0:
            # ignore first 'self' argument for instance methods
            args_msg = ['{}={}'.format(name, arg)
                        for name, arg in zip(kwonly_args[:extra_args],
                                             args[-extra_args:])]
            plural = "s" if len(args_msg) > 1 else ""
            warnings.warn("Pass {} as keyword arg{}. From version 0.12, "
                          "passing these as positional arguments will "
                          "result in an error"
                          .format(", ".join(args_msg), plural),
                          FutureWarning)
        kwargs.update({k: arg for k, arg in zip(sig.parameters, args)})
        return f(**kwargs)
    return inner_f
