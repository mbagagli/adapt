# Contributing guidelines

For the creation of pull request, please follow these suggestions

- Fork the repo.
- Create a new branch and base it **always** at `DEVELOP`.
- Modify the code, and please make sure that you add a test for your change.
Only refactoring and documentation changes require no new tests.
- Once all tests are passed, double-check you have based your pull-request to the `DEVELOP` branch
- Please wait for the reviews and eventual follow-up discussions on your
precious contribution.

The branch name should be representative of what are you actually trying to achieve/improve.
Depending on the improvement you want to achieve, use the following schema
for branch naming:

- `develop_FEATURENAME`: improve the current state of the software.
- `bugfix_FEATURENAME`: to correct / fix previous broken/faulty code features or library dependencies
- `document_FEATURENAME`: to be used when updating the docs or the function's docstring.

As a rule of thumb, we try to adhere to [PEP-8](https://peps.python.org/pep-0008/)
standards and [PEP-257](https://peps.python.org/pep-0257) for Python coding style-guide
and docstring documentation respectively.

For the documentation style guide, though, please refer to the [Google one](https://google.github.io/styleguide/pyguide.html).
Here's an [example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
If possible, always prefer the use of `"` rather than `'` for string coding.

Finally, please refer to the `CODE-OF-CONDUCT` file on how to behave inside the community.
By participating, you are expected to uphold this code. Unacceptable behavior will not be tolerated
and must be reported to the developer.
