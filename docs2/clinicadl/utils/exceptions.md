# Exceptions

[Clinicadl Index](../../README.md#clinicadl-index) /
[Clinicadl](../index.md#clinicadl) /
[Utils](./index.md#utils) /
Exceptions

> Auto-generated documentation for [clinicadl.utils.exceptions](../../../clinicadl/utils/exceptions.py) module.

- [Exceptions](#exceptions)
  - [ClinicaDLArgumentError](#clinicadlargumenterror)
  - [ClinicaDLConfigurationError](#clinicadlconfigurationerror)
  - [ClinicaDLDataLeakageError](#clinicadldataleakageerror)
  - [ClinicaDLException](#clinicadlexception)
  - [ClinicaDLNetworksError](#clinicadlnetworkserror)
  - [ClinicaDLTSVError](#clinicadltsverror)
  - [DownloadError](#downloaderror)
  - [MAPSError](#mapserror)

## ClinicaDLArgumentError

[Show source in exceptions.py:5](../../../clinicadl/utils/exceptions.py#L5)

Base class for ClinicaDL CLI Arguments error.

#### Signature

```python
class ClinicaDLArgumentError(ValueError):
    ...
```



## ClinicaDLConfigurationError

[Show source in exceptions.py:9](../../../clinicadl/utils/exceptions.py#L9)

Base class for ClinicaDL configurations error.

#### Signature

```python
class ClinicaDLConfigurationError(ValueError):
    ...
```



## ClinicaDLDataLeakageError

[Show source in exceptions.py:25](../../../clinicadl/utils/exceptions.py#L25)

Base class for data leakage exceptions.

#### Signature

```python
class ClinicaDLDataLeakageError(ClinicaDLException):
    ...
```

#### See also

- [ClinicaDLException](#clinicadlexception)



## ClinicaDLException

[Show source in exceptions.py:13](../../../clinicadl/utils/exceptions.py#L13)

Base class for ClinicaDL exceptions.

#### Signature

```python
class ClinicaDLException(Exception):
    ...
```



## ClinicaDLNetworksError

[Show source in exceptions.py:21](../../../clinicadl/utils/exceptions.py#L21)

Base class for Networks exceptions.

#### Signature

```python
class ClinicaDLNetworksError(ClinicaDLException):
    ...
```

#### See also

- [ClinicaDLException](#clinicadlexception)



## ClinicaDLTSVError

[Show source in exceptions.py:29](../../../clinicadl/utils/exceptions.py#L29)

Base class for tsv files exceptions.

#### Signature

```python
class ClinicaDLTSVError(ClinicaDLException):
    ...
```

#### See also

- [ClinicaDLException](#clinicadlexception)



## DownloadError

[Show source in exceptions.py:1](../../../clinicadl/utils/exceptions.py#L1)

Base class for download errors exceptions.

#### Signature

```python
class DownloadError(Exception):
    ...
```



## MAPSError

[Show source in exceptions.py:17](../../../clinicadl/utils/exceptions.py#L17)

Base class for MAPS exceptions.

#### Signature

```python
class MAPSError(ClinicaDLException):
    ...
```

#### See also

- [ClinicaDLException](#clinicadlexception)