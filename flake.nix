{
  description = "ragtag dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      forAllSystems = nixpkgs.lib.genAttrs [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          llama-cpp-rpc = (pkgs.llama-cpp.override { rpcSupport = true; }).overrideAttrs (old: {
            # nixpkgs' install phase expects the RPC binary to be called
            # `rpc-server`, but llama.cpp builds it as `ggml-rpc-server`.
            preInstall = (old.preInstall or "") + "\n    ln -s ggml-rpc-server bin/rpc-server\n";
          });
        in
        {
          default = pkgs.mkShell {
            packages = [
              pkgs.rustc
              pkgs.cargo
              pkgs.clippy
              pkgs.rustfmt
              pkgs.rust-analyzer

              # gRPC codegen
              pkgs.protobuf

              # C/C++ build tooling (for llama.cpp, ring, etc.)
              pkgs.gcc
              pkgs.gnumake
              pkgs.cmake
              pkgs.ninja
              pkgs.pkg-config
              pkgs.perl

              # llama.cpp RPC server + llama-cli coordinator
              llama-cpp-rpc

              pkgs.llvmPackages.libclang

              # Crypto / TLS native deps
              pkgs.openssl
            ];

            shellHook = ''
              export PROTOC="${pkgs.protobuf}/bin/protoc"
              export PROTOC_INCLUDE="${pkgs.protobuf}/include"
              export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
              export CMAKE_GENERATOR="Ninja"
              echo "ragtag dev shell ready :D"
            '';
          };
        }
      );
    };
}
