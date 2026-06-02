//! Persistent homology: filtration, barcode computation, persistence diagrams, Betti numbers.

use crate::complex::Simplex;
use serde::{Serialize, Deserialize};

/// A persistence pair: birth and death of a topological feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistencePair {
    pub dim: usize,
    pub birth: f64,
    pub death: f64,
}

impl PersistencePair {
    pub fn new(dim: usize, birth: f64, death: f64) -> Self {
        PersistencePair { dim, birth, death }
    }

    /// Persistence (lifetime) of this feature.
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }

    /// Is this a essential feature (never dies)?
    pub fn is_essential(&self) -> bool {
        self.death == f64::INFINITY
    }

    /// Midpoint of the (birth, death) interval.
    pub fn midpoint(&self) -> f64 {
        (self.birth + self.death) / 2.0
    }
}

/// A persistence diagram: collection of persistence pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    pub fn new(pairs: Vec<PersistencePair>) -> Self {
        PersistenceDiagram { pairs }
    }

    pub fn pairs_of_dim(&self, dim: usize) -> Vec<&PersistencePair> {
        self.pairs.iter().filter(|p| p.dim == dim).collect()
    }

    /// Number of pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Maximum persistence across all pairs.
    pub fn max_persistence(&self) -> f64 {
        self.pairs.iter().map(|p| p.persistence()).fold(0.0f64, f64::max)
    }

    /// Total persistence (sum of all persistences).
    pub fn total_persistence(&self) -> f64 {
        self.pairs.iter().map(|p| p.persistence()).sum()
    }
}

/// A filtration of simplicial complexes indexed by parameter value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filtration {
    /// (parameter_value, simplex) pairs, sorted by value then simplex.
    steps: Vec<(f64, Simplex)>,
}

impl Filtration {
    pub fn new() -> Self {
        Filtration { steps: Vec::new() }
    }

    pub fn add(&mut self, value: f64, simplex: Simplex) {
        self.steps.push((value, simplex));
    }

    /// Sort the filtration by value.
    pub fn sort(&mut self) {
        self.steps.sort_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });
    }

    pub fn steps(&self) -> &[(f64, Simplex)] {
        &self.steps
    }

    /// Build a Vietoris-Rips filtration from point data.
    pub fn vietoris_rips_filtration(
        points: &[nalgebra::DVector<f64>],
        max_dim: usize,
    ) -> Self {
        use crate::complex::pairwise_distances;
        let dists = pairwise_distances(points);
        let n = points.len();

        let mut filt = Filtration::new();

        // Add vertices at 0
        for i in 0..n {
            filt.add(0.0, Simplex::vertex(i));
        }

        // Add edges at their distance
        for i in 0..n {
            for j in (i + 1)..n {
                filt.add(dists[(i, j)], Simplex::edge(i, j));
            }
        }

        // Add higher simplices at the max edge distance (Vietoris-Rips convention)
        if max_dim >= 2 {
            use itertools::Itertools;
            for combo in (0..n).combinations(max_dim + 1) {
                let mut max_d = 0.0f64;
                for (ii, &a) in combo.iter().enumerate() {
                    for &b in combo.iter().skip(ii + 1) {
                        max_d = max_d.max(dists[(a, b)]);
                    }
                }
                let simplex = Simplex::new(combo);
                filt.add(max_d, simplex);
            }
        }

        filt.sort();
        filt
    }

    /// Compute persistent homology from this filtration using the matrix reduction algorithm.
    pub fn compute_persistence(&self) -> PersistenceDiagram {
        let n = self.steps.len();
        if n == 0 {
            return PersistenceDiagram::new(vec![]);
        }

        // Build boundary matrix (mod 2)
        let mut boundary: Vec<Vec<usize>> = vec![vec![]; n];
        let simplex_index: std::collections::HashMap<Simplex, usize> = self.steps
            .iter()
            .enumerate()
            .map(|(i, (_, s))| (s.clone(), i))
            .collect();

        for (j, (_, simplex)) in self.steps.iter().enumerate() {
            for face in simplex.faces() {
                if let Some(&i) = simplex_index.get(&face) {
                    boundary[j].push(i);
                }
            }
            boundary[j].sort();
        }

        // Standard reduction algorithm
        let mut low: Vec<Option<usize>> = vec![None; n];
        for j in 0..n {
            if !boundary[j].is_empty() {
                low[j] = Some(*boundary[j].last().unwrap());
            }
        }

        let mut reduced_low = low.clone();
        let mut reduced_boundary = boundary.clone();

        for j in 0..n {
            loop {
                let l = match reduced_low[j] {
                    Some(l) => l,
                    None => break,
                };
                // Find the earliest column with the same low
                let mut found = false;
                for k in 0..j {
                    if reduced_low[k] == Some(l) {
                        // Add column k to column j (symmetric difference)
                        reduced_boundary[j] = symmetric_difference(
                            &reduced_boundary[j], &reduced_boundary[k]
                        );
                        if reduced_boundary[j].is_empty() {
                            reduced_low[j] = None;
                        } else {
                            reduced_boundary[j].sort();
                            reduced_low[j] = Some(*reduced_boundary[j].last().unwrap());
                        }
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
            }
        }

        // Extract persistence pairs
        let mut paired: std::collections::HashSet<usize> = std::collections::HashSet::new();
        let mut pairs: Vec<PersistencePair> = Vec::new();

        for j in 0..n {
            if let Some(l) = reduced_low[j] {
                paired.insert(l);
                paired.insert(j);
                let birth_val = self.steps[l].0;
                let death_val = self.steps[j].0;
                let dim = self.steps[l].1.dim();
                pairs.push(PersistencePair::new(dim, birth_val, death_val));
            }
        }

        // Unpaired columns are essential (infinite persistence)
        for j in 0..n {
            if !paired.contains(&j) {
                let birth_val = self.steps[j].0;
                let dim = self.steps[j].1.dim();
                pairs.push(PersistencePair::new(dim, birth_val, f64::INFINITY));
            }
        }

        PersistenceDiagram::new(pairs)
    }
}

/// Symmetric difference of two sorted vectors.
fn symmetric_difference(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            result.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            result.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    while i < a.len() {
        result.push(a[i]);
        i += 1;
    }
    while j < b.len() {
        result.push(b[j]);
        j += 1;
    }
    result
}

/// Compute persistent homology from a simplicial complex at a given scale.
pub fn compute_persistent_homology(
    points: &[nalgebra::DVector<f64>],
    max_dim: usize,
) -> PersistenceDiagram {
    let filt = Filtration::vietoris_rips_filtration(points, max_dim);
    filt.compute_persistence()
}

/// Compute Betti numbers at a given filtration value.
pub fn betti_numbers(diagram: &PersistenceDiagram, epsilon: f64) -> Vec<usize> {
    let max_dim = diagram.pairs.iter().map(|p| p.dim).max().unwrap_or(0);
    let mut betti = vec![0usize; max_dim + 1];

    for pair in &diagram.pairs {
        if pair.birth <= epsilon && (pair.death > epsilon || pair.is_essential()) {
            if pair.dim < betti.len() {
                betti[pair.dim] += 1;
            }
        }
    }

    betti
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_persistence_pair() {
        let p = PersistencePair::new(1, 0.5, 2.0);
        assert!((p.persistence() - 1.5).abs() < 1e-10);
        assert!(!p.is_essential());
        assert!((p.midpoint() - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_persistence_essential() {
        let p = PersistencePair::new(0, 0.0, f64::INFINITY);
        assert!(p.is_essential());
        assert!(p.persistence().is_infinite());
    }

    #[test]
    fn test_persistence_diagram_properties() {
        let pairs = vec![
            PersistencePair::new(0, 0.0, 1.0),
            PersistencePair::new(0, 0.0, f64::INFINITY),
            PersistencePair::new(1, 0.5, 2.0),
        ];
        let dg = PersistenceDiagram::new(pairs);
        assert_eq!(dg.len(), 3);
        assert_eq!(dg.pairs_of_dim(0).len(), 2);
        assert_eq!(dg.pairs_of_dim(1).len(), 1);
        assert!((dg.max_persistence() - f64::INFINITY).abs() < 1e-10 || dg.max_persistence().is_infinite());
    }

    #[test]
    fn test_total_persistence() {
        let pairs = vec![
            PersistencePair::new(0, 0.0, 1.0),
            PersistencePair::new(1, 0.5, 2.0),
        ];
        let dg = PersistenceDiagram::new(pairs);
        assert!((dg.total_persistence() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_filtration_single_point() {
        let points = vec![DVector::from_vec(vec![0.0, 0.0])];
        let filt = Filtration::vietoris_rips_filtration(&points, 2);
        let dg = filt.compute_persistence();
        // One vertex → one essential H0 feature
        assert_eq!(dg.pairs.len(), 1);
        assert_eq!(dg.pairs[0].dim, 0);
        assert!(dg.pairs[0].is_essential());
    }

    #[test]
    fn test_filtration_two_points() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
        ];
        let filt = Filtration::vietoris_rips_filtration(&points, 1);
        let dg = filt.compute_persistence();
        // Two vertices, one edge → one H0 essential pair, one H0 born-and-dies
        assert!(dg.pairs.iter().any(|p| p.dim == 0 && p.is_essential()));
    }

    #[test]
    fn test_betti_numbers_single_point() {
        let points = vec![DVector::from_vec(vec![0.0, 0.0])];
        let dg = compute_persistent_homology(&points, 2);
        let bn = betti_numbers(&dg, 0.0);
        assert_eq!(bn[0], 1); // One connected component
    }

    #[test]
    fn test_betti_numbers_triangle() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let dg = compute_persistent_homology(&points, 1);
        // At large epsilon, should have beta0 >= 1
        let bn = betti_numbers(&dg, 2.0);
        assert!(bn[0] >= 1);
    }

    #[test]
    fn test_symmetric_difference() {
        let a = vec![0, 2, 4];
        let b = vec![1, 2, 3];
        let sd = symmetric_difference(&a, &b);
        assert_eq!(sd, vec![0, 1, 3, 4]);
    }

    #[test]
    fn test_symmetric_difference_empty() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2, 3];
        let sd = symmetric_difference(&a, &b);
        assert!(sd.is_empty());
    }

    #[test]
    fn test_filtration_three_collinear() {
        let points = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![2.0]),
        ];
        let dg = compute_persistent_homology(&points, 1);
        // Three collinear points: H0 should have one essential feature
        let h0_essential = dg.pairs_of_dim(0).iter().filter(|p| p.is_essential()).count();
        assert_eq!(h0_essential, 1);
    }

    #[test]
    fn test_diagram_serialization() {
        let pairs = vec![
            PersistencePair::new(0, 0.0, 1.0),
            PersistencePair::new(1, 0.5, 3.0),
        ];
        let dg = PersistenceDiagram::new(pairs);
        let json = serde_json::to_string(&dg).unwrap();
        let dg2: PersistenceDiagram = serde_json::from_str(&json).unwrap();
        assert_eq!(dg2.pairs.len(), 2);
    }
}
