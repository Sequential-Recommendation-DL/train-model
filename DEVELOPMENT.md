# Development Environment Setup

## Environment Setup

### Using Direnv (Recommended)
This project uses [direnv](https://direnv.net/) to automatically load the development environment when you enter the project directory.

- Install direnv if you haven't already:
    ```bash
    # macOS
    brew install direnv

    # Linux (Debian/Ubuntu)
    sudo apt-get install direnv

    # Linux (Fedora)
    sudo dnf install direnv
    ```

- Hook direnv into your shell (add to your ~/.zshrc or ~/.bashrc):
    ```bash
    eval "$(direnv hook zsh)"  # for zsh
    eval "$(direnv hook bash)" # for bash
    ```

- Allow the environment for this project:
    ```bash
    direnv allow
    ```
    This will automatically load the flake.nix configuration and provide you with a shell containing all required dependencies.

### Using Nix Develop Directly
If you prefer not to use direnv, you can manually enter the development environment:

```bash
nix develop
```

This will drop you into a shell with:
- Python 3.12
- NumPy
- Pandas
- PyTorch
- TorchVision
- Matplotlib

## Generating requirements.txt
If you need a traditional requirements.txt file (for example, to deploy to a non-Nix environment), you can generate it from the flake:

```bash
# Method 1: Using nix develop and pip freeze
nix develop
pip freeze > requirements.txt

# Method 2: Directly from flake (alternative approach)
nix build -f flake.nix '#devShells.x86_64-linux.default'
# Then extract packages from the result (more complex)
```

Note: The flake.nix already defines exact versions, so the generated requirements.txt will be consistent.

## Getting Started as a Developer

### Running the Code
The project structure follows a typical machine learning layout:
- `src/data/`: Data loading and preprocessing
- `src/features/`: Feature engineering
- `src/models/`: Model definitions (NeuMF, GMF, MLP)
- `src/pipeline/`: Training and evaluation pipelines

To run the training pipeline:
```bash
python -m src.pipeline.pipeline
```

### Testing
Currently, the project doesn't have a formal test suite. As you develop features, consider adding tests in a `tests/` directory using pytest.

### Linting and Code Quality
The development environment includes:
- [ruff](https://github.com/astral-sh/ruff) for fast linting
- [black](https://github.com/psf/black) for code formatting
- [mypy](https://mypy-lang.org/) for type checking

You can run these tools directly:
```bash
# Linting with ruff
ruff check .

# Code formatting with black
black .

# Type checking with mypy
mypy src/
```

Example configuration in pyproject.toml:
```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.black]
line-length = 88
target-version = ["py312"]

[tool.mypy]
python_version = "3.12"
```

## Troubleshooting

### Common Issues
- **direnv not loading**: Ensure you've hooked direnv into your shell and ran `direnv allow`
- **Permission errors with nix**: Make sure you have Nix installed with flake support enabled
- **Package conflicts**: The flake pins specific versions; avoid mixing with system Python packages

### Getting Help
- Consult the [Nix documentation](https://nixos.org/manual/nix/stable/)
- Check direnv troubleshooting: https://direnv.net/
- For PyTorch issues: https://pytorch.org/get-started/locally/
