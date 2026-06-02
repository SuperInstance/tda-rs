//! # tda-rs
//!
//! Topological data analysis (TDA) — extracting shape from data.
//!
//! Provides simplicial complexes, persistent homology, Betti numbers,
//! bottleneck/Wasserstein distances, persistence landscapes, the Mapper algorithm,
//! nerve theorem verification, and statistical TDA.

#![deny(unsafe_code)]
#![allow(clippy::needless_range_loop, clippy::redundant_closure, clippy::while_let_loop, clippy::collapsible_if, clippy::ptr_arg, clippy::new_without_default)]

pub mod complex;
pub mod persistence;
pub mod distance;
pub mod landscape;
pub mod mapper;
pub mod nerve;
pub mod statistics;

pub use complex::{Simplex, SimplicialComplex, VietorisRips, AlphaComplex, CechComplex};
pub use persistence::{Filtration, PersistenceDiagram, PersistencePair, compute_persistent_homology, betti_numbers};
pub use distance::{bottleneck_distance, wasserstein_distance};
pub use landscape::{PersistenceLandscape, landscape_distance};
pub use mapper::{Mapper, MapperGraph};
pub use nerve::{verify_nerve_theorem, Cover, NerveVerification};
pub use statistics::{bootstrap_persistence, confidence_set};

