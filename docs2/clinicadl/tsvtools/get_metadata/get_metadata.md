# Get Metadata

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Tsvtools](../index.md#tsvtools) /
[Get Metadata](./index.md#get-metadata) /
Get Metadata

> Auto-generated documentation for [clinicadl.tsvtools.get_metadata.get_metadata](../../../../clinicadl/tsvtools/get_metadata/get_metadata.py) module.

- [Get Metadata](#get-metadata)
  - [get_metadata](#get_metadata)

## get_metadata

[Show source in get_metadata.py:13](../../../../clinicadl/tsvtools/get_metadata/get_metadata.py#L13)

Get the meta data in metadata_df to write them in output_df.
If variables_of_interest is None, the function writes all the data that are in metadata_df for the list of subjects in output_df.

Parameters
----------
data_tsv: str (Path)
    Columns must include ['participant_id', 'session_id']
merged_tsv: str (Path)
    output of `clinica merge-tsv`
variables_of_interest: list of str
    List of columns that will be added in the output DataFrame.

Returns
-------

#### Signature

```python
def get_metadata(
    data_tsv: Path, merged_tsv: Path, variables_of_interest=None
) -> pd.DataFrame:
    ...
```