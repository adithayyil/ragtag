use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::Path;
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "xtask")]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Run `cargo clippy --all-targets --all-features -- -D warnings`.
    Clippy,
    /// Run `cargo test --all-features`.
    Test,
    /// Run `cargo fmt`.
    Fmt,
    /// Run `cargo fmt --check`.
    FmtCheck,
}

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest is inside the workspace")
}

fn run(cmd: &str, args: &[&str]) -> Result<()> {
    let status = Command::new(cmd)
        .current_dir(workspace_root())
        .args(args)
        .status()
        .with_context(|| format!("failed to execute {cmd} {:?}", args))?;
    if !status.success() {
        bail!("{} {:?} exited with {}", cmd, args, status);
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.cmd {
        Cmd::Clippy => run(
            "cargo",
            &[
                "clippy",
                "--all-targets",
                "--all-features",
                "--",
                "-D",
                "warnings",
            ],
        ),
        Cmd::Test => run("cargo", &["test", "--all-features"]),
        Cmd::Fmt => run("cargo", &["fmt"]),
        Cmd::FmtCheck => run("cargo", &["fmt", "--check"]),
    }
}
