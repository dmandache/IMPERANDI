To build the documentation, run these commands from the repository root:

```bash
python -m pip install -e .
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` in a browser.
