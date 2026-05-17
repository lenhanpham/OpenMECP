# Installation

## Prerequisites

- **Rust 1.70 or later** — install from [rustup.rs](https://rustup.rs)
- **One or more QM programs**: Gaussian 16 (recommended) or ORCA 5+

## Building from Source

```bash
# Clone the repository
git clone https://github.com/lenhanpham/OpenMECP.git
cd OpenMECP

# Build optimized release binary
cargo build --release

# The binary is placed at:
#   target/release/omecp
```

## Installing the Binary

Copy the binary to a directory on your `PATH`:

```bash
# Linux / macOS
cp target/release/omecp ~/.local/bin/
chmod 700 ~/.local/bin/omecp

# Or to a system-wide location (requires sudo)
sudo cp target/release/omecp /usr/local/bin/
```

On HPC systems, copy the binary to a personal `bin/` directory and export it
in your shell profile or job script:

```bash
export PATH="$HOME/bin:$PATH"
```

## Verifying the Installation

```bash
omecp --help
```

You should see the OpenMECP help output listing all available options and topics.

## HPC Setup

On HPC clusters, add the following to your job submission script:

```bash
# Load your QM program module
module load gaussian/g16   # or: module load orca/5.0

# Export the directory containing omecp
export PATH="/path/to/your/bin:$PATH"

# Run the calculation
omecp input.inp > output.log
```

## Building Documentation Locally

```bash
# Install mdBook
cargo install mdbook

# Build and serve the documentation
mdbook serve docs
```

Open `http://localhost:3000` in your browser.

## Building API Reference Locally

```bash
cargo doc --no-deps --open
```
