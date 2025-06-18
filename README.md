# Template for a project with pytest

After cloning the repository, install the project in "editable" mode for development:

```
pip install -e .[test]
```

The `[test]` not only installs the package itself but also dependencies required
for testing. Any changes in the code will automatically be updated in the installed
development package.

After installation, run the tests by submitting

```
pytest
```

The kata is described in detail in [kata-diff.pdf](kata-diff.pdf).
