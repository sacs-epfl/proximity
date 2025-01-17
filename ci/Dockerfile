FROM rustlang/rust:nightly

RUN apt update -y && apt upgrade -y && apt install -y valgrind && rustup component add clippy rustfmt && cargo install cargo-valgrind
