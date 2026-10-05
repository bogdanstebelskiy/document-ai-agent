# Python

Python functions can accept positional and keyword arguments. Default argument values are evaluated when the function is defined, not each time the function is called. This is why mutable defaults such as lists can cause surprising behavior.

List comprehensions are useful for building a list from an iterable in one expression. A generator expression looks similar but produces values lazily, which is useful when the whole result does not need to stay in memory.

A context manager is normally used with `with`. It makes setup and cleanup explicit, such as opening a file and making sure it gets closed afterward. Python's `dataclass` decorator is useful for classes whose main purpose is holding data.

Type hints do not normally enforce types at runtime. They describe intended types and allow tools such as type checkers and IDEs to catch mistakes.
