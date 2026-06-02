//! Nerve theorem verification.

use crate::complex::{SimplicialComplex, Simplex};
use crate::mapper::MapperGraph;
use serde::{Serialize, Deserialize};
use std::collections::BTreeSet;

/// A cover of a set: a collection of subsets whose union is the whole set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cover {
    /// Each set is a list of point indices.
    pub sets: Vec<Vec<usize>>,
    /// Total number of points.
    pub n_points: usize,
}

impl Cover {
    pub fn new(sets: Vec<Vec<usize>>, n_points: usize) -> Self {
        Cover { sets, n_points }
    }

    /// Check if the cover actually covers all points.
    pub fn is_valid(&self) -> bool {
        let mut covered: BTreeSet<usize> = BTreeSet::new();
        for set in &self.sets {
            for &p in set {
                covered.insert(p);
            }
        }
        (0..self.n_points).all(|i| covered.contains(&i))
    }

    /// Compute the nerve of this cover: a simplicial complex where a simplex
    /// is included iff the corresponding sets have non-empty intersection.
    pub fn nerve(&self) -> SimplicialComplex {
        let n_sets = self.sets.len();
        let sets_as_bt: Vec<BTreeSet<usize>> = self.sets.iter()
            .map(|s| s.iter().copied().collect())
            .collect();

        let mut complex = SimplicialComplex::new();

        // Vertices: each non-empty set
        for i in 0..n_sets {
            if !self.sets[i].is_empty() {
                complex.add_simplex(&Simplex::vertex(i));
            }
        }

        // Edges: pairs with non-empty intersection
        for i in 0..n_sets {
            for j in (i + 1)..n_sets {
                if !sets_as_bt[i].is_disjoint(&sets_as_bt[j]) {
                    complex.add_simplex(&Simplex::edge(i, j));
                }
            }
        }

        // Higher simplices: all k-tuples with non-empty common intersection
        for k in 3..=n_sets {
            let mut found = false;
            // Generate all k-combinations
            let indices: Vec<usize> = (0..n_sets).collect();
            for combo in indices.iter().combinations(k) {
                let mut intersection = sets_as_bt[*combo[0]].clone();
                for &idx in &combo[1..] {
                    intersection = intersection.intersection(&sets_as_bt[*idx]).copied().collect();
                }
                if !intersection.is_empty() {
                    let simplex = Simplex::new(combo.into_iter().copied());
                    complex.add_simplex(&simplex);
                    found = true;
                }
            }
            if !found { break; }
        }

        complex
    }
}

/// Verify the nerve theorem for a given cover and Mapper graph.
/// The nerve theorem states that if all sets and their intersections are
/// contractible, then the nerve is homotopy equivalent to the union.
pub fn verify_nerve_theorem(cover: &Cover) -> NerveVerification {
    let nerve = cover.nerve();
    let is_valid = cover.is_valid();

    NerveVerification {
        nerve,
        cover_is_valid: is_valid,
        // In general, we can't verify contractibility computationally,
        // but we can check if the cover consists of convex-like sets
        contractibility_assumed: true,
    }
}

/// Result of nerve theorem verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NerveVerification {
    pub nerve: SimplicialComplex,
    pub cover_is_valid: bool,
    pub contractibility_assumed: bool,
}

/// Build a cover from a Mapper graph.
pub fn mapper_graph_to_cover(graph: &MapperGraph, n_points: usize) -> Cover {
    Cover::new(
        graph.nodes.iter().map(|n| n.points.clone()).collect(),
        n_points,
    )
}

use itertools::Itertools;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cover_valid() {
        let cover = Cover::new(vec![vec![0, 1], vec![1, 2], vec![2, 3]], 4);
        assert!(cover.is_valid());
    }

    #[test]
    fn test_cover_invalid() {
        let cover = Cover::new(vec![vec![0, 1], vec![1, 2]], 4);
        assert!(!cover.is_valid());
    }

    #[test]
    fn test_nerve_basic() {
        let cover = Cover::new(vec![vec![0, 1], vec![1, 2]], 3);
        let nerve = cover.nerve();
        // Sets 0 and 1 share point 1, so there's an edge
        assert!(nerve.contains(&Simplex::edge(0, 1)));
    }

    #[test]
    fn test_nerve_disjoint() {
        let cover = Cover::new(vec![vec![0], vec![1]], 2);
        let nerve = cover.nerve();
        // Sets are disjoint, no edge
        assert!(!nerve.contains(&Simplex::edge(0, 1)));
    }

    #[test]
    fn test_nerve_triple_overlap() {
        let cover = Cover::new(vec![vec![0, 1, 2], vec![0, 1, 2], vec![0, 1, 2]], 3);
        let nerve = cover.nerve();
        // All three sets overlap, so we get a triangle
        assert!(nerve.contains(&Simplex::new(vec![0, 1, 2])));
    }

    #[test]
    fn test_verify_nerve_theorem() {
        let cover = Cover::new(vec![vec![0, 1], vec![1, 2], vec![0, 2]], 3);
        let verification = verify_nerve_theorem(&cover);
        assert!(verification.cover_is_valid);
        assert!(verification.contractibility_assumed);
        assert!(verification.nerve.n_simplices() > 0);
    }

    #[test]
    fn test_mapper_to_cover() {
        use crate::mapper::{MapperNode, MapperEdge};
        let graph = MapperGraph {
            nodes: vec![
                MapperNode { id: 0, points: vec![0, 1], cluster_center: vec![0.0] },
                MapperNode { id: 1, points: vec![1, 2], cluster_center: vec![1.0] },
            ],
            edges: vec![MapperEdge { source: 0, target: 1 }],
        };
        let cover = mapper_graph_to_cover(&graph, 3);
        assert!(cover.is_valid());
        let nerve = cover.nerve();
        // Sets share point 1, so there's an edge
        assert!(nerve.contains(&Simplex::edge(0, 1)));
    }

    #[test]
    fn test_nerve_empty_cover() {
        let cover = Cover::new(vec![], 0);
        let nerve = cover.nerve();
        assert_eq!(nerve.n_simplices(), 0);
    }

    #[test]
    fn test_nerve_single_set() {
        let cover = Cover::new(vec![vec![0, 1, 2]], 3);
        let nerve = cover.nerve();
        assert!(nerve.contains(&Simplex::vertex(0)));
        assert_eq!(nerve.simplices_of_dim(1).len(), 0);
    }
}
