# Development Environment Setup

There are two ways to set up the development environment:

## Method 1: Using Flake (Recommended for Nix users)

1. Ensure you have [Nix](https://nixos.org/download.html) installed with flake support.
2. From the project root, run:
   ```bash
   nix develop
   ```
3. This will drop you into a shell with the required dependencies.

## Method 2: Using venv

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate environment:
   - Windows (CMD): `venv\Scripts\activate`
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - Linux/macOS: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. If requirements.txt is not present, you can generate it from the flake or by using:

   ```bash
   pip freeze > requirements.txt
   ```

   (Note: The flake.nix already defines the dependencies, so you can also create requirements.txt by translating the flake.)

5. To deactivate the environment:

   ```bash
   deactivate

   ```

6. Explicit Dataset

- https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
