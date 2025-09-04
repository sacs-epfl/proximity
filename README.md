# Proximity 

Proximity is a research project exploring the optimization and speed-recall tradeoffs of approximate vector search in high-dimensional spaces.
We provide an approximate cache for vector databases that is written in Rust and exposes Python bindings.

More information is available in our main [findings paper](https://arxiv.org/abs/2503.05530), currently under review. It expands on our [EuroMLSys '25 publication](https://dl.acm.org/doi/10.1145/3721146.3721941).

Note: This code is under active development and is not recommended for production systems.

## Installation

### Prerequisites

Ensure you have the following dependencies installed:

- the Rust toolkit (Cargo and rustup are encouraged). For now, nightly is required, but a regular (non-nightly) install on your machine will automatically download the nightly compiler and use it only in this project, making this effectively transparent for the user.
- Python 3.8+
- Maturin for Rust-Python interactions (we recommend installing by running ```pip install maturin```)

## Build Instructions

``` 
python3 -m venv proxi-env
source proxi-env/bin/activate
git clone https://gitlab.epfl.ch/randl/proximity.git
cd proximity/bindings
maturin develop -r
```

## Usage

todo

## Repository Structure

```proximity/
├── bindings/       # Python bindings
├── core/           # Rust source code
├── ci/             # Continuous integration build scripts
├── README.md
└── LICENSE         # MIT License
```
## License

This project is licensed under the MIT License. See LICENSE for details.

