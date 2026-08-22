# LocalPilot

## Features

- Add SQLite-backed machine knowledge and change history

## Usage

### Storing Machine Facts

```python
import machine_knowledge

machine_knowledge.store_fact('os', 'Windows')
```

### Recording Changes

```python
import machine_knowledge

machine_knowledge.record_change('os', 'Windows', 'Linux')
```
