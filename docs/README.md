# Documentation

This directory contains data related to Sphinx doc generation. To generate basic
docs, run:

```
pip install -r requirements-doc.txt
make html
sphinx-build -b rinoh source _build/rinoh
```

**Do not** commit generated HTML files to version control.

See [the Sphinx docs](https://www.sphinx-doc.org/en/master/tutorial/index.html)
for more information.
