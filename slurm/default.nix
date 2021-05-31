with import (builtins.fetchGit {
  name = "nixpkgs-2020-11-11";
  url = "https://github.com/NixOS/nixpkgs/";
  rev = "dd1b7e377f6d77ddee4ab84be11173d3566d6a18";
}) { config = { allowUnfree = true; }; };


mkShell {
  name = "beste-shell";
  venvDir = "./_venv";
  # Add dependencies that pip can't fetch here (or that we don't want to
  # install using pip).
  buildInputs = (with pkgs.python38Packages; [ python numpy venvShellHook wheel ])
    ++ (import ./system-dependencies.nix { inherit pkgs; });
  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    pip install -r suprb2/requirements.txt
  '';

}
