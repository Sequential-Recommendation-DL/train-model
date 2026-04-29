{
  description = "Development environment configuration for NeuMF";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (pkgs.python312.withPackages (
            ps: with ps; [
              numpy
              pandas
              pytorch
              matplotlib
            ]
          ))
        ];

        shellHook = ''
          echo "Development environment configuration ready!" 
          echo "Python: $(python --version)"
        '';

      };
    };
}
