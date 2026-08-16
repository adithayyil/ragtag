use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?);
    let proto_dir = manifest.join("proto");
    let proto = proto_dir.join("ragtag.proto");

    println!("cargo:rerun-if-changed=proto/ragtag.proto");

    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&[&proto], &[&proto_dir])?;

    Ok(())
}
