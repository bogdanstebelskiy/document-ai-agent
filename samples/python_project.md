# Python Project Structure

A `src` layout keeps the importable package separate from project-level files such as tests and configuration. A typical project might have `src/my_package`, `tests`, `pyproject.toml`, and a lock file.

`pyproject.toml` is the standard place for Python project metadata and tool configuration. Modern projects can use it to declare dependencies, build settings, and configuration for tools such as pytest and Ruff.

An application configuration object can centralize settings such as model names, database paths, and chunk sizes. Environment variables are useful for values that differ between machines.

Dependency injection can make components easier to test. Instead of constructing a database or vector store inside every class, a component can receive the dependency in its constructor and tests can provide a fake implementation.
