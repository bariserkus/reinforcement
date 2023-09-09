# Python program that can be executed to report whether particular
# python packages are available on the system.

import math
import os
import sys


def test_is_python_35():
    major = sys.version_info.major
    minor = sys.version_info.minor
    if major == 3:
        pass
    else:
        print("You are running Python {}, but we need Python {}.".format(major, 3))
        print("Download and install the Anaconda distribution for Python 3.")
        print("Stopping here.")

        # Let's stop here
        sys.exit(1)
        # assert major == 3, "Stopping here - we need Python 3."

    if minor >= 5:
        print("Testing Python version-> py{}.{} OK".format(major, minor))
    else:
        print(
            "Warning: You should be running Python 3.5 or newer, "
            + "you have Python {}.{}.".format(major, minor)
        )


def test_numpy():
    try:
        import numpy as np
    except ImportError:
        print("Could not import numpy -> numpy failed")
        return None
    # Simple test
    a = np.arange(0, 100, 1)
    assert np.sum(a) == sum(a)
    print("Testing numpy...      -> numpy OK")


def test_scipy():
    try:
        import scipy
    except ImportError:
        print("Could not import 'scipy' -> scipy failed")
        return None
    # Simple test
    import scipy.integrate

    assert abs(scipy.integrate.quad(lambda x: x * x, 0, 6)[0] - 72.0) < 1e-6
    print("Testing scipy ...     -> scipy OK")


"""
def test_matplotlib():
    try:
        import matplotlib as plt
        import numpy as np
    except ImportError:
        print("Could not import 'scipy' -> scipy failed")
        return None
    # Simple test
    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(
        xlabel="time (s)",
        ylabel="voltage (mV)",
        title="About as simple as it gets, folks",
    )
    ax.grid()
    matfilename = "testmatplot.png"

    if os.path.exists(matfilename):
        print(
            "Skipping plotting to file as file {} exists already.".format(matfilename)
        )
    else:
        # Write plot to file
        fig.savefig(matfilename)
        # Then check that file exists
        assert os.path.exists(matfilename)
        print("Testing matplotlib... -> matplotlib OK")
        os.remove(matfilename)

    plt.show()
    print("Testing matplotlib ...     -> matplotlib OK")
"""


def test_pylab():
    """Actually testing matplotlib, as pylab is part of matplotlib."""
    try:
        import pylab
    except ImportError:
        print("Could not import 'matplotlib/pylab' -> failed")
        return None
    # Create plot for testing purposes
    xvalues = [i * 0.1 for i in range(100)]
    yvalues = [math.sin(x) for x in xvalues]
    pylab.plot(xvalues, yvalues, "-o", label="sin(x)")
    pylab.legend()
    pylab.xlabel("x")
    testfilename = "pylab-testfigure.png"

    # check that file does not exist yet:
    if os.path.exists(testfilename):
        print(
            "Skipping plotting to file as file {} exists already.".format(testfilename)
        )
    else:
        # Write plot to file
        pylab.savefig(testfilename)
        # Then check that file exists
        assert os.path.exists(testfilename)
        print("Testing matplotlib... -> pylab OK")
        os.remove(testfilename)


def test_sympy():
    try:
        import sympy
    except ImportError:
        print("Could not import 'sympy' -> fail")
        return None
    # simple test
    x = sympy.Symbol("x")
    my_f = x**2
    assert sympy.diff(my_f, x) == 2 * x
    print("Testing sympy         -> sympy OK")


def test_pytest():
    try:
        import pytest
    except ImportError:
        print("Could not import 'pytest' -> fail")
        return None
    print("Testing pytest        -> pytest OK")


if __name__ == "__main__":
    print("Running using Python {}".format(sys.version))
    test_is_python_35()
    test_numpy()
    test_scipy()
    test_pylab()
    test_sympy()
    test_pytest()
    # test_matplotlib()
