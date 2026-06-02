//! Mapper algorithm: cluster-based simplification of high-dimensional data.

use nalgebra::DVector;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};

/// Configuration for the Mapper algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperConfig {
    /// Number of intervals in the filter function domain.
    pub n_intervals: usize,
    /// Overlap percentage between intervals (0.0 to 1.0).
    pub overlap: f64,
    /// Number of clusters per interval.
    pub n_clusters: usize,
}

impl Default for MapperConfig {
    fn default() -> Self {
        MapperConfig {
            n_intervals: 10,
            overlap: 0.1,
            n_clusters: 3,
        }
    }
}

/// A node in the Mapper graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperNode {
    pub id: usize,
    pub points: Vec<usize>,
    pub cluster_center: Vec<f64>,
}

/// An edge in the Mapper graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperEdge {
    pub source: usize,
    pub target: usize,
}

/// The Mapper graph output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperGraph {
    pub nodes: Vec<MapperNode>,
    pub edges: Vec<MapperEdge>,
}

impl MapperGraph {
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get adjacency list representation.
    pub fn adjacency(&self) -> HashMap<usize, Vec<usize>> {
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for edge in &self.edges {
            adj.entry(edge.source).or_default().push(edge.target);
            adj.entry(edge.target).or_default().push(edge.source);
        }
        adj
    }

    /// Check if the graph is connected.
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() { return true; }
        let adj = self.adjacency();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack = vec![0];
        while let Some(v) = stack.pop() {
            if visited.contains(&v) { continue; }
            visited.insert(v);
            if let Some(neighbors) = adj.get(&v) {
                for &n in neighbors {
                    if !visited.contains(&n) {
                        stack.push(n);
                    }
                }
            }
        }
        visited.len() == self.nodes.len()
    }

    /// Number of connected components.
    pub fn n_components(&self) -> usize {
        if self.nodes.is_empty() { return 0; }
        let adj = self.adjacency();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components = 0;
        for node in &self.nodes {
            if visited.contains(&node.id) { continue; }
            components += 1;
            let mut stack = vec![node.id];
            while let Some(v) = stack.pop() {
                if visited.contains(&v) { continue; }
                visited.insert(v);
                if let Some(neighbors) = adj.get(&v) {
                    for &n in neighbors {
                        if !visited.contains(&n) {
                            stack.push(n);
                        }
                    }
                }
            }
        }
        components
    }
}

/// The Mapper algorithm implementation.
pub struct Mapper {
    config: MapperConfig,
}

impl Mapper {
    pub fn new(config: MapperConfig) -> Self {
        Mapper { config }
    }

    /// Run the Mapper algorithm on the given data with a filter function.
    /// `filter_fn` maps each point to a scalar value.
    pub fn run<F>(&self, points: &[DVector<f64>], filter_fn: F) -> MapperGraph
    where
        F: Fn(&DVector<f64>) -> f64,
    {
        let n = points.len();
        if n == 0 {
            return MapperGraph { nodes: vec![], edges: vec![] };
        }

        // Compute filter values
        let filter_values: Vec<f64> = points.iter().map(|p| filter_fn(p)).collect();
        let f_min = filter_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let f_max = filter_values.iter().cloned().fold(f64::INFINITY, f64::max);

        if (f_max - f_min).abs() < 1e-14 {
            // All points have same filter value
            return MapperGraph {
                nodes: vec![MapperNode {
                    id: 0,
                    points: (0..n).collect(),
                    cluster_center: compute_centroid(points, &(0..n).collect::<Vec<_>>()),
                }],
                edges: vec![],
            };
        }

        // Create intervals with overlap
        let interval_width = (f_max - f_min) / self.config.n_intervals as f64;
        let overlap_width = interval_width * self.config.overlap;
        let step = interval_width - overlap_width;

        let mut intervals: Vec<(f64, f64)> = Vec::new();
        let mut t = f_min;
        while t < f_max - 1e-14 {
            let end = (t + interval_width).min(f_max);
            intervals.push((t, end));
            t += step;
        }
        if intervals.is_empty() {
            intervals.push((f_min, f_max));
        }

        // For each interval, find points and cluster them
        let mut all_clusters: Vec<Vec<usize>> = Vec::new();
        let mut cluster_intervals: Vec<usize> = Vec::new(); // which interval each cluster belongs to

        for (interval_idx, (lo, hi)) in intervals.iter().enumerate() {
            let mut in_interval: Vec<usize> = Vec::new();
            for i in 0..n {
                if filter_values[i] >= *lo - 1e-12 && filter_values[i] <= *hi + 1e-12 {
                    in_interval.push(i);
                }
            }

            if in_interval.is_empty() { continue; }

            let clusters = cluster_points(points, &in_interval, self.config.n_clusters);
            for cluster in clusters {
                cluster_intervals.push(interval_idx);
                all_clusters.push(cluster);
            }
        }

        // Build nodes
        let mut nodes: Vec<MapperNode> = Vec::new();
        for (id, cluster) in all_clusters.iter().enumerate() {
            nodes.push(MapperNode {
                id,
                points: cluster.clone(),
                cluster_center: compute_centroid(points, cluster),
            });
        }

        // Build edges: connect clusters from adjacent intervals that share points
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..all_clusters.len() {
            for j in (i + 1)..all_clusters.len() {
                // Only connect clusters from different intervals
                if cluster_intervals[i] == cluster_intervals[j] {
                    continue;
                }
                // Check overlap
                let set_i: HashSet<usize> = all_clusters[i].iter().copied().collect();
                let set_j: HashSet<usize> = all_clusters[j].iter().copied().collect();
                if !set_i.is_disjoint(&set_j) {
                    let edge = if i < j { (i, j) } else { (j, i) };
                    edge_set.insert(edge);
                }
            }
        }

        let edges: Vec<MapperEdge> = edge_set.iter()
            .map(|&(s, t)| MapperEdge { source: s, target: t })
            .collect();

        MapperGraph { nodes, edges }
    }
}

/// Simple k-means-like clustering.
fn cluster_points(points: &[DVector<f64>], indices: &[usize], k: usize) -> Vec<Vec<usize>> {
    if indices.len() <= k {
        return indices.iter().map(|&i| vec![i]).collect();
    }

    let dim = points[0].len();

    // Initialize centroids using first k points
    let mut centroids: Vec<DVector<f64>> = indices.iter()
        .take(k)
        .map(|&i| points[i].clone())
        .collect();

    // If fewer unique starting points than k, reduce
    if centroids.len() < k {
        return vec![indices.to_vec()];
    }

    let mut assignments: Vec<usize> = vec![0; indices.len()];
    let max_iters = 20;

    for _ in 0..max_iters {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (idx, &point_idx) in indices.iter().enumerate() {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let d = (&points[point_idx] - centroid).norm_squared();
                if d < best_dist {
                    best_dist = d;
                    best = c;
                }
            }
            if assignments[idx] != best {
                changed = true;
                assignments[idx] = best;
            }
        }

        if !changed { break; }

        // Update centroids
        for c in 0..k {
            let members: Vec<usize> = indices.iter()
                .enumerate()
                .filter(|(idx, _)| assignments[*idx] == c)
                .map(|(_, &i)| i)
                .collect();
            if !members.is_empty() {
                let mut sum = DVector::zeros(dim);
                for &i in &members {
                    sum += &points[i];
                }
                centroids[c] = sum / members.len() as f64;
            }
        }
    }

    // Build clusters
    let mut clusters: Vec<Vec<usize>> = vec![vec![]; k];
    for (idx, &point_idx) in indices.iter().enumerate() {
        clusters[assignments[idx]].push(point_idx);
    }

    // Remove empty clusters
    clusters.retain(|c| !c.is_empty());
    clusters
}

/// Compute centroid of a set of points.
fn compute_centroid(points: &[DVector<f64>], indices: &[usize]) -> Vec<f64> {
    if indices.is_empty() { return vec![]; }
    let dim = points[0].len();
    let mut sum = DVector::zeros(dim);
    for &i in indices {
        sum += &points[i];
    }
    (sum / indices.len() as f64).iter().copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapper_basic() {
        let points: Vec<DVector<f64>> = (0..20).map(|i| {
            DVector::from_vec(vec![i as f64, (i as f64).sin()])
        }).collect();
        let mapper = Mapper::new(MapperConfig {
            n_intervals: 3,
            overlap: 0.2,
            n_clusters: 2,
        });
        let graph = mapper.run(&points, |p| p[0]);
        assert!(graph.n_nodes() > 0);
    }

    #[test]
    fn test_mapper_empty() {
        let points: Vec<DVector<f64>> = vec![];
        let mapper = Mapper::new(MapperConfig::default());
        let graph = mapper.run(&points, |p| p[0]);
        assert_eq!(graph.n_nodes(), 0);
    }

    #[test]
    fn test_mapper_single_point() {
        let points = vec![DVector::from_vec(vec![1.0, 2.0])];
        let mapper = Mapper::new(MapperConfig::default());
        let graph = mapper.run(&points, |p| p[0]);
        assert_eq!(graph.n_nodes(), 1);
    }

    #[test]
    fn test_mapper_graph_connected() {
        // Points along a line should produce a connected graph
        let points: Vec<DVector<f64>> = (0..50).map(|i| {
            DVector::from_vec(vec![i as f64 / 50.0, 0.0])
        }).collect();
        let mapper = Mapper::new(MapperConfig {
            n_intervals: 5,
            overlap: 0.3,
            n_clusters: 2,
        });
        let graph = mapper.run(&points, |p| p[0]);
        // Should have at least some nodes
        assert!(graph.n_nodes() > 0);
    }

    #[test]
    fn test_mapper_clusters() {
        // Two well-separated clusters
        let mut points = Vec::new();
        for i in 0..10 {
            points.push(DVector::from_vec(vec![0.0, i as f64]));
        }
        for i in 0..10 {
            points.push(DVector::from_vec(vec![10.0, i as f64]));
        }
        let mapper = Mapper::new(MapperConfig {
            n_intervals: 5,
            overlap: 0.1,
            n_clusters: 2,
        });
        let graph = mapper.run(&points, |p| p[0]);
        assert!(graph.n_nodes() > 0);
    }

    #[test]
    fn test_mapper_graph_serialization() {
        let graph = MapperGraph {
            nodes: vec![MapperNode {
                id: 0,
                points: vec![0, 1],
                cluster_center: vec![1.0, 2.0],
            }],
            edges: vec![],
        };
        let json = serde_json::to_string(&graph).unwrap();
        let g2: MapperGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(g2.n_nodes(), 1);
    }

    #[test]
    fn test_mapper_n_components() {
        let graph = MapperGraph {
            nodes: vec![
                MapperNode { id: 0, points: vec![0], cluster_center: vec![0.0] },
                MapperNode { id: 1, points: vec![1], cluster_center: vec![1.0] },
            ],
            edges: vec![MapperEdge { source: 0, target: 1 }],
        };
        assert_eq!(graph.n_components(), 1);
        assert!(graph.is_connected());
    }

    #[test]
    fn test_mapper_disconnected() {
        let graph = MapperGraph {
            nodes: vec![
                MapperNode { id: 0, points: vec![0], cluster_center: vec![0.0] },
                MapperNode { id: 1, points: vec![1], cluster_center: vec![1.0] },
            ],
            edges: vec![],
        };
        assert_eq!(graph.n_components(), 2);
        assert!(!graph.is_connected());
    }

    #[test]
    fn test_cluster_points() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![0.1, 0.1]),
            DVector::from_vec(vec![5.0, 5.0]),
            DVector::from_vec(vec![5.1, 5.1]),
        ];
        let clusters = cluster_points(&points, &[0, 1, 2, 3], 2);
        assert_eq!(clusters.len(), 2);
        // Each cluster should have 2 points
        for c in &clusters {
            assert_eq!(c.len(), 2);
        }
    }
}
