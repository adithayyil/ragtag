{
  description = "ragtag dev shell + container image";

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
            preInstall =
              (old.preInstall or "")
              + "\n    ln -s ggml-rpc-server bin/rpc-server\n    ln -s ggml-rpc-server bin/llama-rpc-server\n";
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

      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          llama-cpp-rpc = (pkgs.llama-cpp.override { rpcSupport = true; }).overrideAttrs (old: {
            preInstall =
              (old.preInstall or "")
              + "\n    ln -s ggml-rpc-server bin/rpc-server\n    ln -s ggml-rpc-server bin/llama-rpc-server\n";
          });

          ragtag = pkgs.rustPlatform.buildRustPackage {
            pname = "ragtag";
            version = "0.1.0";

            src = pkgs.lib.fileset.toSource {
              root = ./.;
              fileset = pkgs.lib.fileset.unions [
                ./Cargo.toml
                ./Cargo.lock
                ./crates
                ./xtask
              ];
            };

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            nativeBuildInputs = [
              pkgs.protobuf
              pkgs.pkg-config
              pkgs.cmake
              pkgs.perl
              pkgs.llvmPackages.libclang
            ];

            buildInputs = [
              pkgs.openssl
            ];

            env = {
              PROTOC = "${pkgs.protobuf}/bin/protoc";
              PROTOC_INCLUDE = "${pkgs.protobuf}/include";
              LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
            };

            doCheck = false;

            installPhase = ''
              runHook preInstall
              mkdir -p $out/bin
              cp target/${pkgs.stdenv.hostPlatform.config}/release/ragtag-coordinator $out/bin/
              cp target/${pkgs.stdenv.hostPlatform.config}/release/ragtag-node $out/bin/
              cp target/${pkgs.stdenv.hostPlatform.config}/release/ragtag-chat $out/bin/
              runHook postInstall
            '';

            meta = {
              description = "ragtag distributed LLM inference cluster";
              license = pkgs.lib.licenses.mit;
              platforms = pkgs.lib.platforms.all;
            };
          };

          isLinux = pkgs.stdenv.hostPlatform.isLinux;
        in
        {
          inherit llama-cpp-rpc ragtag;
          default = ragtag;

          containerImage =
            if isLinux then
              pkgs.dockerTools.buildLayeredImage {
                name = "ragtag";
                tag = "latest";
                created = "now";

                contents = [
                  ragtag
                  llama-cpp-rpc
                  pkgs.cacert
                  pkgs.bash
                  pkgs.coreutils
                ];

                config = {
                  Entrypoint = [ "/bin/ragtag-coordinator" ];
                  ExposedPorts = {
                    "3000/tcp" = { };
                    "50051/tcp" = { };
                    "50052/tcp" = { };
                    "8081/tcp" = { };
                  };
                  Env = [
                    "SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt"
                  ];
                };
              }
            else
              throw "containerImage is only supported on Linux";
        }
      );
    };
}
