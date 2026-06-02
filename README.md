# tda-rs

> **The only TDA library in Rust.** Find the shape of your data.

Topological Data Analysis in Rust — persistent homology, simplicial complexes, Mapper algorithm, persistence landscapes, Betti curves, bottleneck and Wasserstein distances, bootstrap confidence.

## Quick Start

```rust
use tda_rs::*;
use nalgebra::DVector;

// Point cloud
let points: Vec<DVector<f64>> = vec![
    DVector::from_vec(vec![0.0, 0.0]),
    DVector::from_vec(vec![1.0, 0.0]),
    DVector::from_vec(vec![0.0, 1.0]),
    DVector::from_vec(vec![1.0, 1.0]),
];

// Compute persistent homology (Vietoris-Rips filtration)
let diagram = persistence::compute_persistent_homology(&points, 2);

// Read off topological features
for pair in &diagram.pairs {
    println!("H{}: born {:.3}, died {:.3} (persistence {:.3})",
        pair.dim, pair.birth, pair.death, pair.persistence());
}

// Betti numbers at a given scale
let betti = persistence::betti_numbers(&diagram, 1.0);
println!("Betti numbers at ε=1.0: {:?}", betti);

// Persistence landscape
let landscape = landscape::PersistenceLandscape::from_diagram(&diagram);
println!("Landscape layers: {}", landscape.num_layers());

// Compare two datasets
let d2 = persistence::compute_persistent_homology(&points, 2);
let dist = distance::bottleneck_distance(&diagram, &d2);
println!("Bottleneck distance: {:.3}", dist);
```

## What's Inside

| Module | Description |
|--------|-------------|
| `complex` | Vietoris-Rips, Čech, and Alpha complexes |
| `persistence` | Persistent homology via matrix reduction over Z₂ |
| `distance` | Bottleneck and Wasserstein distances between diagrams |
| `landscape` | Persistence landscapes with Lᵖ norms and integration |
| `mapper` | Mapper algorithm for topological simplification |
| `nerve` | Nerve theorem verification for covers |
| `statistics` | Bootstrap resampling and confidence sets |

## Install

```toml
[dependencies]
tda-rs = "0.1"
```

## License

MIT OR Apache-2.0
