import pathlib


def test_run_examples(example_filename):
    print(example_filename)
    if pathlib.Path(example_filename).name == "__init__.py":
        return
    with open(example_filename, encoding="utf8") as f:
        exec(f.read(), {"__name__": "__main__"})  # pylint: disable=exec-used
