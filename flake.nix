{
  description = "lycklig - image stacking and correction of atmospheric distortion";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        lycklig = pkgs.callPackage ./lycklig.nix { };

      in {
        packages = {
          default = lycklig;
          lycklig = lycklig;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [ self.packages.${system}.default ];
          buildInputs = with pkgs; [
            gdb
            valgrind
            clang-tools
          ];

          shellHook = ''
            echo
            echo "lycklig development environment"
            echo
            echo "C++ compiler: $(which gcc)"
            echo "CMake: $(which cmake)"
            echo "Python: $(which python3)"
            echo
            echo "Don't forget to wrapQtApp kinky after building manually!"
            echo
          '';
        };

        apps = {
          default = flake-utils.lib.mkApp {
            drv = lycklig;
          };
          lycklig = flake-utils.lib.mkApp {
            drv = lycklig;
          };
          kinky = flake-utils.lib.mkApp {
            drv = lycklig;
            exePath = "/bin/kinky";
          };
        };
      });
}
