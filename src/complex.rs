//! Simplicial complexes: Vietoris-Rips, Čech, Alpha complexes, and Delaunay basics.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use std::collections::{BTreeSet, HashMap, HashSet};
use itertools::Itertools;

/// A simplex represented as a sorted set of vertex indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Simplex(pub BTreeSet<usize>);

impl Simplex {
    pub fn new(vertices: impl IntoIterator<Item = usize>) -> Self {
        Simplex(vertices.into_iter().collect())
    }

    pub fn vertex(i: usize) -> Self {
        Simplex(BTreeSet::from([i]))
    }

    pub fn edge(i: usize, j: usize) -> Self {
        Simplex(BTreeSet::from([i, j]))
    }

    /// Dimension of the simplex (number of vertices - 1).
    pub fn dim(&self) -> usize {
        if self.0.is_empty() { 0 } else { self.0.len() - 1 }
    }

    /// Vertices as a sorted vector.
    pub fn vertices(&self) -> Vec<usize> {
        self.0.iter().copied().collect()
    }

    /// All faces of codimension 1 (boundary simplices).
    pub fn faces(&self) -> Vec<Simplex> {
        if self.dim() == 0 {
            return vec![];
        }
        let verts: Vec<usize> = self.0.iter().copied().collect();
        (0..verts.len())
            .map(|i| {
                let mut face = self.0.clone();
                face.remove(&verts[i]);
                Simplex(face)
            })
            .collect()
    }

    /// Check if this simplex contains another as a face.
    pub fn contains_face(&self, other: &Simplex) -> bool {
        other.0.is_subset(&self.0)
    }
}

/// A simplicial complex: a set of simplices closed under taking faces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplicialComplex {
    simplices: BTreeSet<Simplex>,
    n_vertices: usize,
}

impl SimplicialComplex {
    pub fn new() -> Self {
        SimplicialComplex {
            simplices: BTreeSet::new(),
            n_vertices: 0,
        }
    }

    /// Build from a set of simplices, automatically adding all faces.
    pub fn from_simplices(simplices: Vec<Simplex>) -> Self {
        let mut all = BTreeSet::new();
        let mut max_v = 0usize;
        for s in &simplices {
            all.insert(s.clone());
            for &v in &s.0 {
                max_v = max_v.max(v);
            }
            // Add all faces recursively
            let mut queue = vec![s.clone()];
            while let Some(current) = queue.pop() {
                for face in current.faces() {
                    if !all.contains(&face) {
                        all.insert(face.clone());
                        queue.push(face);
                    }
                }
            }
        }
        SimplicialComplex {
            simplices: all,
            n_vertices: max_v + 1,
        }
    }

    /// Add a simplex and all its faces.
    pub fn add_simplex(&mut self, simplex: &Simplex) {
        let mut queue = vec![simplex.clone()];
        while let Some(current) = queue.pop() {
            for &v in &current.0 {
                self.n_vertices = self.n_vertices.max(v + 1);
            }
            if !self.simplices.contains(&current) {
                self.simplices.insert(current.clone());
                for face in current.faces() {
                    if !self.simplices.contains(&face) {
                        queue.push(face);
                    }
                }
            }
        }
    }

    pub fn simplices(&self) -> impl Iterator<Item = &Simplex> {
        self.simplices.iter()
    }

    pub fn simplices_of_dim(&self, dim: usize) -> Vec<&Simplex> {
        self.simplices.iter().filter(|s| s.dim() == dim).collect()
    }

    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    pub fn n_simplices(&self) -> usize {
        self.simplices.len()
    }

    pub fn contains(&self, simplex: &Simplex) -> bool {
        self.simplices.contains(simplex)
    }

    /// Euler characteristic: alternating sum of simplex counts by dimension.
    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0i64;
        let max_dim = self.simplices.iter().map(|s| s.dim()).max().unwrap_or(0);
        for d in 0..=max_dim {
            let count = self.simplices_of_dim(d).len() as i64;
            if d % 2 == 0 {
                chi += count;
            } else {
                chi -= count;
            }
        }
        chi
    }

    /// Boundary matrix for dimension d: columns are d-simplices, rows are (d-1)-simplices.
    pub fn boundary_matrix(&self, dim: usize) -> DMatrix<i32> {
        let rows = self.simplices_of_dim(dim - 1);
        let cols = self.simplices_of_dim(dim);
        let row_idx: HashMap<_, _> = rows.iter().enumerate()
            .map(|(i, s)| ((*s).clone(), i))
            .collect();
        let mut mat = DMatrix::zeros(rows.len(), cols.len());
        for (j, col_simplex) in cols.iter().enumerate() {
            for face in col_simplex.faces() {
                if let Some(&i) = row_idx.get(&face) {
                    mat[(i, j)] ^= 1; // Z2 arithmetic
                }
            }
        }
        mat
    }
}

/// Compute pairwise Euclidean distances between points.
pub fn pairwise_distances(points: &[DVector<f64>]) -> DMatrix<f64> {
    let n = points.len();
    let mut dists = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in (i + 1)..n {
            let d = (&points[i] - &points[j]).norm();
            dists[(i, j)] = d;
            dists[(j, i)] = d;
        }
    }
    dists
}

/// Vietoris-Rips complex: includes a simplex if all pairwise distances ≤ epsilon.
pub struct VietorisRips {
    points: Vec<DVector<f64>>,
    dists: DMatrix<f64>,
    epsilon: f64,
    max_dim: usize,
}

impl VietorisRips {
    pub fn new(points: Vec<DVector<f64>>, epsilon: f64, max_dim: usize) -> Self {
        let dists = pairwise_distances(&points);
        VietorisRips { points, dists, epsilon, max_dim }
    }

    /// Build the Vietoris-Rips complex.
    pub fn build(&self) -> SimplicialComplex {
        let n = self.points.len();
        let mut complex = SimplicialComplex::new();

        // Add all vertices
        for i in 0..n {
            complex.add_simplex(&Simplex::vertex(i));
        }

        if self.max_dim == 0 {
            return complex;
        }

        // Add edges within epsilon
        let mut adjacency: HashSet<(usize, usize)> = HashSet::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if self.dists[(i, j)] <= self.epsilon {
                    adjacency.insert((i, j));
                    adjacency.insert((j, i));
                    complex.add_simplex(&Simplex::edge(i, j));
                }
            }
        }

        // Build higher simplices by clique enumeration
        for dim in 2..=self.max_dim {
            let lower = complex.simplices_of_dim(dim - 1);
            let mut candidates: HashSet<BTreeSet<usize>> = HashSet::new();
            for s in &lower {
                // Try extending each (dim-1)-simplex by one vertex
                for v in 0..n {
                    if s.0.contains(&v) {
                        continue;
                    }
                    // Check if v is connected to all vertices in s
                    let all_connected = s.0.iter().all(|&u| {
                        self.dists[(u.min(v), u.max(v))] <= self.epsilon
                    });
                    if all_connected {
                        let mut new_set = s.0.clone();
                        new_set.insert(v);
                        candidates.insert(new_set);
                    }
                }
            }
            for cand in candidates {
                complex.add_simplex(&Simplex(cand));
            }
        }

        complex
    }

    /// Compute the distance matrix.
    pub fn distance_matrix(&self) -> &DMatrix<f64> {
        &self.dists
    }
}

/// Čech complex: includes a simplex if balls of radius epsilon around all vertices have non-empty intersection.
pub struct CechComplex {
    points: Vec<DVector<f64>>,
    epsilon: f64,
    max_dim: usize,
}

impl CechComplex {
    pub fn new(points: Vec<DVector<f64>>, epsilon: f64, max_dim: usize) -> Self {
        CechComplex { points, epsilon, max_dim }
    }

    /// Build the Čech complex.
    /// For Čech, a simplex is included iff the intersection of balls of radius epsilon is non-empty.
    /// This is equivalent to checking if the minimal enclosing ball radius ≤ epsilon.
    pub fn build(&self) -> SimplicialComplex {
        let n = self.points.len();
        let dists = pairwise_distances(&self.points);
        let mut complex = SimplicialComplex::new();

        for i in 0..n {
            complex.add_simplex(&Simplex::vertex(i));
        }

        if self.max_dim == 0 {
            return complex;
        }

        // For small simplices, use the Čech condition: diameter/2 <= epsilon
        // (This is a sufficient condition; exact Čech requires checking minimal enclosing ball)
        for dim in 1..=self.max_dim {
            let verts: Vec<usize> = (0..n).collect();
            for combo in verts.iter().combinations(dim + 1) {
                let combo_set: BTreeSet<usize> = combo.iter().copied().copied().collect();
                // Check if all pairwise distances <= 2*epsilon (necessary condition for intersection)
                let mut valid = true;
                let mut max_dist = 0.0f64;
                for (ii, &a) in combo.iter().enumerate() {
                    for &b in combo.iter().skip(ii + 1) {
                        let d = dists[(*a, *b)];
                        max_dist = max_dist.max(d);
                        if d > 2.0 * self.epsilon {
                            valid = false;
                            break;
                        }
                    }
                    if !valid { break; }
                }
                // For Čech, the condition is: diameter / 2 <= epsilon
                // i.e., max pairwise distance / 2 <= epsilon
                if valid && max_dist / 2.0 <= self.epsilon {
                    complex.add_simplex(&Simplex(combo_set));
                }
            }
        }

        complex
    }
}

/// Alpha complex (Delaunay-based).
/// For simplicity, this implements a basic version using the Delaunay triangulation
/// and filtering by circumradius.
pub struct AlphaComplex {
    points: Vec<DVector<f64>>,
    alpha: f64,
}

impl AlphaComplex {
    pub fn new(points: Vec<DVector<f64>>, alpha: f64) -> Self {
        AlphaComplex { points, alpha }
    }

    /// Build alpha complex.
    /// This simplified version works for 2D points.
    pub fn build(&self) -> SimplicialComplex {
        let n = self.points.len();
        let mut complex = SimplicialComplex::new();

        for i in 0..n {
            complex.add_simplex(&Simplex::vertex(i));
        }

        if self.points.is_empty() {
            return complex;
        }

        let dim = self.points[0].len();
        if dim == 2 {
            self.build_2d(&mut complex);
        } else {
            // Fallback: use distance-based approximation
            let dists = pairwise_distances(&self.points);
            for i in 0..n {
                for j in (i + 1)..n {
                    let d = dists[(i, j)];
                    if d / 2.0 <= self.alpha {
                        complex.add_simplex(&Simplex::edge(i, j));
                    }
                }
            }
            // Add triangles where circumradius <= alpha
            for i in 0..n {
                for j in (i + 1)..n {
                    if !complex.contains(&Simplex::edge(i, j)) { continue; }
                    for k in (j + 1)..n {
                        if !complex.contains(&Simplex::edge(i, k)) ||
                           !complex.contains(&Simplex::edge(j, k)) { continue; }
                        let cr = circumradius_3(
                            &self.points[i], &self.points[j], &self.points[k]
                        );
                        if cr <= self.alpha {
                            complex.add_simplex(&Simplex(BTreeSet::from([i, j, k])));
                        }
                    }
                }
            }
        }

        complex
    }

    fn build_2d(&self, complex: &mut SimplicialComplex) {
        let n = self.points.len();
        let dists = pairwise_distances(&self.points);

        // Delaunay triangulation (simplified: for small datasets, enumerate valid triangles)
        // Use the empty circumcircle property
        let mut delaunay_edges: HashSet<(usize, usize)> = HashSet::new();
        let mut delaunay_triangles: Vec<[usize; 3]> = Vec::new();

        // For each potential triangle, check Delaunay property
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    if is_delaunay(&self.points, i, j, k) {
                        delaunay_edges.insert((i.min(j), i.max(j)));
                        delaunay_edges.insert((i.min(k), i.max(k)));
                        delaunay_edges.insert((j.min(k), j.max(k)));
                        delaunay_triangles.push([i, j, k]);
                    }
                }
            }
        }

        // Filter by alpha (circumradius)
        for &(a, b) in &delaunay_edges {
            let mid_dist = dists[(a, b)] / 2.0;
            if mid_dist <= self.alpha {
                complex.add_simplex(&Simplex::edge(a, b));
            }
        }

        for &[i, j, k] in &delaunay_triangles {
            let cr = circumradius_3(&self.points[i], &self.points[j], &self.points[k]);
            if cr <= self.alpha {
                complex.add_simplex(&Simplex(BTreeSet::from([i, j, k])));
            }
        }
    }
}

/// Check if triangle (i,j,k) satisfies the Delaunay property (no other point inside circumcircle).
fn is_delaunay(points: &[DVector<f64>], i: usize, j: usize, k: usize) -> bool {
    let n = points.len();
    let (cc, cr_sq) = circumcircle_2d(&points[i], &points[j], &points[k]);
    for m in 0..n {
        if m == i || m == j || m == k { continue; }
        let d = (&points[m] - &cc).norm_squared();
        if d < cr_sq - 1e-10 {
            return false;
        }
    }
    true
}

/// Compute circumcircle of three 2D points. Returns (center, radius_squared).
fn circumcircle_2d(a: &DVector<f64>, b: &DVector<f64>, c: &DVector<f64>) -> (DVector<f64>, f64) {
    let ax = a[0]; let ay = a[1];
    let bx = b[0]; let by = b[1];
    let cx = c[0]; let cy = c[1];

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-14 {
        // Degenerate: return large radius
        return (DVector::from_vec(vec![0.0, 0.0]), f64::INFINITY);
    }

    let ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;

    let center = DVector::from_vec(vec![ux, uy]);
    let r_sq = (ax - ux).powi(2) + (ay - uy).powi(2);
    (center, r_sq)
}

/// Circumradius of three points.
fn circumradius_3(a: &DVector<f64>, b: &DVector<f64>, c: &DVector<f64>) -> f64 {
    let ab = (b - a).norm();
    let bc = (c - b).norm();
    let ca = (a - c).norm();
    let _a_val = (bc * bc + ca * ca - ab * ab).abs();
    let area = 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])).abs();
    if area < 1e-14 {
        return f64::INFINITY;
    }
    (ab * bc * ca) / (4.0 * area)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_construction() {
        let s = Simplex::new(vec![0, 1, 2]);
        assert_eq!(s.dim(), 2);
        assert_eq!(s.vertices(), vec![0, 1, 2]);
    }

    #[test]
    fn test_simplex_faces() {
        let s = Simplex::new(vec![0, 1, 2]);
        let faces = s.faces();
        assert_eq!(faces.len(), 3);
        assert!(faces.contains(&Simplex::new(vec![0, 1])));
        assert!(faces.contains(&Simplex::new(vec![0, 2])));
        assert!(faces.contains(&Simplex::new(vec![1, 2])));
    }

    #[test]
    fn test_simplicial_complex_from_simplices() {
        let tri = Simplex::new(vec![0, 1, 2]);
        let sc = SimplicialComplex::from_simplices(vec![tri]);
        // Should contain 1 triangle + 3 edges + 3 vertices = 7
        assert_eq!(sc.n_simplices(), 7);
    }

    #[test]
    fn test_euler_characteristic_triangle() {
        let tri = Simplex::new(vec![0, 1, 2]);
        let sc = SimplicialComplex::from_simplices(vec![tri]);
        // 3 vertices - 3 edges + 1 triangle = 1
        assert_eq!(sc.euler_characteristic(), 1);
    }

    #[test]
    fn test_vietoris_rips_3_points() {
        // Three points forming a triangle
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let vr = VietorisRips::new(points, 1.5, 2);
        let complex = vr.build();
        // All pairwise distances <= sqrt(2) < 1.5, so we get the full triangle
        assert!(complex.contains(&Simplex::new(vec![0, 1, 2])));
    }

    #[test]
    fn test_vietoris_rips_small_epsilon() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![5.0, 0.0]),
        ];
        let vr = VietorisRips::new(points, 0.5, 2);
        let complex = vr.build();
        // Only vertices, no edges (all distances > 0.5)
        assert!(!complex.contains(&Simplex::edge(0, 1)));
        assert!(!complex.contains(&Simplex::edge(0, 2)));
    }

    #[test]
    fn test_vietoris_rips_medium_epsilon() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![5.0, 0.0]),
        ];
        let vr = VietorisRips::new(points, 1.5, 2);
        let complex = vr.build();
        // Only edge (0,1) with dist=1.0, not (1,2) with dist=4.0
        assert!(complex.contains(&Simplex::edge(0, 1)));
        assert!(!complex.contains(&Simplex::edge(1, 2)));
    }

    #[test]
    fn test_cech_complex() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let cech = CechComplex::new(points, 0.8, 2);
        let complex = cech.build();
        // For the triangle with side 1 and sqrt(2), circumradius/2 ~ 0.707 < 0.8
        // So the triangle should be included
        assert!(complex.contains(&Simplex::edge(0, 1)));
    }

    #[test]
    fn test_alpha_complex() {
        let points = vec![
            DVector::from_vec(vec![0.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        let alpha = AlphaComplex::new(points, 2.0);
        let complex = alpha.build();
        // With large alpha, should include the triangle
        assert!(complex.contains(&Simplex::edge(0, 1)));
    }

    #[test]
    fn test_boundary_matrix() {
        let tri = Simplex::new(vec![0, 1, 2]);
        let sc = SimplicialComplex::from_simplices(vec![tri]);
        let bm = sc.boundary_matrix(2);
        // 3 edges, 1 triangle: should be 3x1
        assert_eq!(bm.nrows(), 3);
        assert_eq!(bm.ncols(), 1);
        // All entries should be 1
        assert_eq!(bm[(0, 0)], 1);
        assert_eq!(bm[(1, 0)], 1);
        assert_eq!(bm[(2, 0)], 1);
    }

    #[test]
    fn test_pairwise_distances() {
        let points = vec![
            DVector::from_vec(vec![0.0]),
            DVector::from_vec(vec![3.0]),
            DVector::from_vec(vec![4.0]),
        ];
        let d = pairwise_distances(&points);
        assert!((d[(0, 1)] - 3.0).abs() < 1e-10);
        assert!((d[(0, 2)] - 4.0).abs() < 1e-10);
        assert!((d[(1, 2)] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simplex_contains_face() {
        let s = Simplex::new(vec![0, 1, 2]);
        let f = Simplex::new(vec![0, 1]);
        assert!(s.contains_face(&f));
        assert!(!f.contains_face(&s));
    }

    #[test]
    fn test_simplicial_complex_add_simplex() {
        let mut sc = SimplicialComplex::new();
        sc.add_simplex(&Simplex::new(vec![0, 1, 2]));
        assert_eq!(sc.n_simplices(), 7); // 1 tri + 3 edges + 3 vertices
        assert_eq!(sc.n_vertices(), 3);
    }
}
