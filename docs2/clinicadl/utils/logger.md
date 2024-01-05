# Logger

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Utils](./index.md#utils) /
Logger

> Auto-generated documentation for [clinicadl.utils.logger](../../../clinicadl/utils/logger.py) module.

- [Logger](#logger)
  - [ConsoleFormatter](#consoleformatter)
    - [ConsoleFormatter().format](#consoleformatter()format)
  - [StdLevelFilter](#stdlevelfilter)
    - [StdLevelFilter().filter](#stdlevelfilter()filter)
  - [setup_logging](#setup_logging)

## ConsoleFormatter

[Show source in logger.py:18](../../../clinicadl/utils/logger.py#L18)

#### Signature

```python
class ConsoleFormatter(logging.Formatter):
    ...
```

### ConsoleFormatter().format

[Show source in logger.py:25](../../../clinicadl/utils/logger.py#L25)

#### Signature

```python
def format(self, record):
    ...
```



## StdLevelFilter

[Show source in logger.py:6](../../../clinicadl/utils/logger.py#L6)

#### Signature

```python
class StdLevelFilter(logging.Filter):
    def __init__(self, err=False):
        ...
```

### StdLevelFilter().filter

[Show source in logger.py:11](../../../clinicadl/utils/logger.py#L11)

#### Signature

```python
def filter(self, record):
    ...
```



## setup_logging

[Show source in logger.py:31](../../../clinicadl/utils/logger.py#L31)

Setup ClinicaDL's logging facilities.

#### Arguments

- `verbose` - The desired level of verbosity for logging.
    - `(False` *default* - INFO, True: DEBUG)

#### Signature

```python
def setup_logging(verbose: bool = False) -> None:
    ...
```