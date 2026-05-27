{
  description = "Development environment configuration for NeuMF";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true; # required for NVIDIA CUDA packages
          cudaSupport = true;
        };
      };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (pkgs.python312.withPackages (
            ps: with ps; [
              numpy
              pandas
              torch
              torchvision
              matplotlib
              ruff
              black
              mypy
              pip
              huggingface-hub
              orjson
            ]
          ))
        ];

        shellHook = ''
          if [ -f .env ]; then
            export $(grep -v '^#' .env | xargs)
          fi
          echo "Development environment configuration ready!" 
          echo "Python: $(python --version)"
        '';

      };
    };
}
