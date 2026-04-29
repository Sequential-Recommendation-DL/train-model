# Virtual Environment (venv) Guide

## 1. Create virtual environment

```bash
python -m venv venv
```

## 2. Activate environment

### Windows (CMD)

```bash
venv\Scripts\activate
```

### Windows (PowerShell)

```bash
venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

- Create requirements.txt if not available:

```bash
pip freeze > requirements.txt
```
## 4. Deactivate environment

```bash
deactivate
```
