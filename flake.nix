{
  description = "Development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          python312
          python312Packages.pip
          python312Packages.virtualenv
          stdenv.cc.cc.lib
          zlib
        ];

        shellHook = ''
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:/run/opengl-driver/lib:$LD_LIBRARY_PATH"

          if [ -f .env ]; then
            export $(grep -v '^#' .env | xargs)
          fi

          if [ ! -d .venv ]; then
            python -m venv .venv
          fi
          source .venv/bin/activate

          if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "Installing CUDA-enabled PyTorch (cu128)..."
            pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu128
            pip install --quiet numpy pandas matplotlib scikit-learn tqdm ruff black mypy huggingface-hub orjson
          fi

          echo "Development environment configuration ready!"
          echo "Python: $(python --version)"
          python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
        '';

      };
    };
}
