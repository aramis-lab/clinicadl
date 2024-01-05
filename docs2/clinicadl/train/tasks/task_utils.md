# Task Utils

[Clinicadl Index](../../../README.md#clinicadl-index) /
[Clinicadl](../../index.md#clinicadl) /
[Train](../index.md#train) /
[Tasks](./index.md#tasks) /
Task Utils

> Auto-generated documentation for [clinicadl.train.tasks.task_utils](../../../../clinicadl/train/tasks/task_utils.py) module.

- [Task Utils](#task-utils)
  - [task_launcher](#task_launcher)

## task_launcher

[Show source in task_utils.py:8](../../../../clinicadl/train/tasks/task_utils.py#L8)

Common training framework for all tasks

#### Arguments

- `network_task` - task learnt by the network.
- `task_options_list` - list of options specific to the task.
- `kwargs` - other arguments and options for network training.

#### Signature

```python
def task_launcher(network_task: str, task_options_list: List[str], **kwargs):
    ...
```