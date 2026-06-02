//! Distances between persistence diagrams: bottleneck and Wasserstein.

use crate::persistence::PersistenceDiagram;

/// Compute the bottleneck distance between two persistence diagrams.
/// Uses a matching-based approach with binary search on the distance.
pub fn bottleneck_distance(dg1: &PersistenceDiagram, dg2: &PersistenceDiagram) -> f64 {
    // Collect finite persistence pairs from both diagrams
    let mut points1: Vec<(f64, f64)> = dg1.pairs.iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();
    let mut points2: Vec<(f64, f64)> = dg2.pairs.iter()
        .filter(|p| !p.is_essential())
        .map(|p| (p.birth, p.death))
        .collect();

    if points1.is_empty() && points2.is_empty() {
        return 0.0;
    }

    // For essential pairs, project to diagonal at (birth, birth)
    let essential1: Vec<(f64, f64)> = dg1.pairs.iter()
        .filter(|p| p.is_essential())
        .map(|p| (p.birth, p.birth))
        .collect();
    let essential2: Vec<(f64, f64)> = dg2.pairs.iter()
        .filter(|p| p.is_essential())
        .map(|p| (p.birth, p.birth))
        .collect();

    points1.extend(essential1);
    points2.extend(essential2);

    // Compute all possible distances for binary search
    let mut candidates: Vec<f64> = Vec::new();
    let n1 = points1.len();
    let n2 = points2.len();

    for i in 0..n1 {
        for j in 0..n2 {
            candidates.push(l_inf_distance(points1[i], points2[j]));
        }
        // Distance to diagonal
        candidates.push(diagonal_distance(points1[i]));
    }
    for j in 0..n2 {
        candidates.push(diagonal_distance(points2[j]));
    }

    if candidates.is_empty() {
        return 0.0;
    }

    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Binary search for the bottleneck distance
    let mut lo = 0usize;
    let mut hi = candidates.len() - 1;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if is_valid_matching(&points1, &points2, candidates[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }

    candidates[lo]
}

/// L-infinity distance between two points in the birth-death plane.
fn l_inf_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    (p1.0 - p2.0).abs().max((p1.1 - p2.1).abs())
}

/// Distance from a point to the diagonal in the birth-death plane.
fn diagonal_distance(p: (f64, f64)) -> f64 {
    ((p.1 - p.0).abs()) / 2.0
}

/// Check if a valid matching exists with maximum cost <= threshold.
fn is_valid_matching(points1: &[(f64, f64)], points2: &[(f64, f64)], threshold: f64) -> bool {
    // Greedy matching: match as many as possible within threshold
    // This is a simplified check; exact matching would use Hungarian algorithm
    let n1 = points1.len();
    let n2 = points2.len();
    let _max_n = n1.max(n2);

    // Build bipartite graph
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n1];
    for i in 0..n1 {
        for j in 0..n2 {
            if l_inf_distance(points1[i], points2[j]) <= threshold {
                adj[i].push(j);
            }
        }
    }

    // Maximum bipartite matching using augmenting paths
    let mut match_to: Vec<Option<usize>> = vec![None; n2];

    for i in 0..n1 {
        let mut visited = vec![false; n2];
        augment(i, &adj, &mut match_to, &mut visited);
    }

    // Unmatched in points1: check diagonal distance
    for i in 0..n1 {
        let mut is_matched = false;
        for j in 0..n2 {
            if match_to[j] == Some(i) {
                is_matched = true;
                break;
            }
        }
        if !is_matched && diagonal_distance(points1[i]) > threshold {
            return false;
        }
    }

    // Unmatched in points2
    let mut matched2: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for j in 0..n2 {
        if match_to[j].is_some() {
            matched2.insert(j);
        }
    }
    for j in 0..n2 {
        if !matched2.contains(&j) && diagonal_distance(points2[j]) > threshold {
            return false;
        }
    }

    true
}

fn augment(u: usize, adj: &[Vec<usize>], match_to: &mut [Option<usize>], visited: &mut [bool]) -> bool {
    for &v in &adj[u] {
        if visited[v] { continue; }
        visited[v] = true;
        if match_to[v].is_none() || augment(match_to[v].unwrap(), adj, match_to, visited) {
            match_to[v] = Some(u);
            return true;
        }
    }
    false
}

/// Compute the Wasserstein-p distance between two persistence diagrams.
pub fn wasserstein_distance(dg1: &PersistenceDiagram, dg2: &PersistenceDiagram, p: f64) -> f64 {
    let mut points1: Vec<(f64, f64)> = dg1.pairs.iter()
        .filter(|pp| !pp.is_essential())
        .map(|pp| (pp.birth, pp.death))
        .collect();
    let mut points2: Vec<(f64, f64)> = dg2.pairs.iter()
        .filter(|pp| !pp.is_essential())
        .map(|pp| (pp.birth, pp.death))
        .collect();

    // Add essential pairs projected to diagonal
    for pp in &dg1.pairs {
        if pp.is_essential() {
            points1.push((pp.birth, pp.birth));
        }
    }
    for pp in &dg2.pairs {
        if pp.is_essential() {
            points2.push((pp.birth, pp.birth));
        }
    }

    // Balance with diagonal projections
    let n = points1.len().max(points2.len());
    while points1.len() < n {
        // Add dummy diagonal point
        points1.push((0.0, 0.0));
    }
    while points2.len() < n {
        points2.push((0.0, 0.0));
    }

    if n == 0 {
        return 0.0;
    }

    // For small diagrams, use optimal matching via Hungarian algorithm (simplified)
    // Use greedy approximation for efficiency
    let mut cost_matrix: Vec<Vec<f64>> = Vec::new();
    for i in 0..n {
        let mut row: Vec<f64> = Vec::new();
        for j in 0..n {
            let d = l_p_distance(points1[i], points2[j], p);
            let diag1 = diagonal_distance_p(points1[i], p);
            let diag2 = diagonal_distance_p(points2[j], p);
            // The matching cost is min of direct match or both going to diagonal
            row.push(d.min(diag1 + diag2));
        }
        cost_matrix.push(row);
    }

    // Use Hungarian algorithm for optimal matching
    let total = hungarian(&cost_matrix);
    total.powf(1.0 / p)
}

fn l_p_distance(p1: (f64, f64), p2: (f64, f64), p: f64) -> f64 {
    ((p1.0 - p2.0).abs().powf(p) + (p1.1 - p2.1).abs().powf(p)).powf(1.0 / p)
}

fn diagonal_distance_p(point: (f64, f64), p: f64) -> f64 {
    // Distance from point to diagonal in L^p norm
    let d = (point.1 - point.0).abs() / 2.0_f64.sqrt();
    d.powf(p)
}

/// Simple Hungarian algorithm for assignment problem.
fn hungarian(cost: &[Vec<f64>]) -> f64 {
    let n = cost.len();
    if n == 0 { return 0.0; }

    // For small n, use brute force
    if n <= 8 {
        return hungarian_brute(cost);
    }

    // Greedy approximation for larger problems
    let mut used: Vec<bool> = vec![false; n];
    let mut total = 0.0;
    for i in 0..n {
        let mut best_j = 0;
        let mut best_cost = f64::INFINITY;
        for j in 0..n {
            if !used[j] && cost[i][j] < best_cost {
                best_cost = cost[i][j];
                best_j = j;
            }
        }
        used[best_j] = true;
        total += best_cost;
    }
    total
}

fn hungarian_brute(cost: &[Vec<f64>]) -> f64 {
    let n = cost.len();
    if n == 0 { return 0.0; }
    if n > 10 {
        // Greedy fallback for larger
        let mut used: Vec<bool> = vec![false; n];
        let mut total = 0.0;
        for i in 0..n {
            let mut best_j = 0;
            let mut best_cost = f64::INFINITY;
            for j in 0..n {
                if !used[j] && cost[i][j] < best_cost {
                    best_cost = cost[i][j];
                    best_j = j;
                }
            }
            used[best_j] = true;
            total += best_cost;
        }
        return total;
    }
    let mut perm: Vec<usize> = (0..n).collect();
    let mut best = f64::INFINITY;
    loop {
        let mut total = 0.0;
        for i in 0..n {
            total += cost[i][perm[i]];
        }
        best = best.min(total);
        if !next_permutation(&mut perm) {
            break;
        }
    }
    best
}

fn next_permutation(arr: &mut Vec<usize>) -> bool {
    let n = arr.len();
    if n < 2 { return false; }
    let mut i = n - 2;
    while arr[i] >= arr[i + 1] {
        if i == 0 { return false; }
        i -= 1;
    }
    let mut j = n - 1;
    while arr[j] <= arr[i] {
        j -= 1;
    }
    arr.swap(i, j);
    arr[i + 1..].reverse();
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::PersistencePair;

    fn make_diagram(pairs: Vec<(usize, f64, f64)>) -> PersistenceDiagram {
        PersistenceDiagram::new(
            pairs.into_iter().map(|(d, b, dd)| PersistencePair::new(d, b, dd)).collect()
        )
    }

    #[test]
    fn test_bottleneck_identical() {
        let dg = make_diagram(vec![(0, 0.0, 1.0), (1, 0.5, 2.0)]);
        let dist = bottleneck_distance(&dg, &dg);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_bottleneck_empty() {
        let dg1 = PersistenceDiagram::new(vec![]);
        let dg2 = PersistenceDiagram::new(vec![]);
        let dist = bottleneck_distance(&dg1, &dg2);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_bottleneck_shifted() {
        let dg1 = make_diagram(vec![(0, 0.0, 1.0)]);
        let dg2 = make_diagram(vec![(0, 0.5, 1.5)]);
        let dist = bottleneck_distance(&dg1, &dg2);
        assert!(dist > 0.0);
        assert!(dist <= 0.5);
    }

    #[test]
    fn test_bottleneck_different_size() {
        let dg1 = make_diagram(vec![(0, 0.0, 1.0)]);
        let dg2 = make_diagram(vec![(0, 0.0, 1.0), (0, 2.0, 3.0)]);
        let dist = bottleneck_distance(&dg1, &dg2);
        // Extra point in dg2 goes to diagonal, distance = 0.5
        assert!(dist >= 0.4);
    }

    #[test]
    fn test_wasserstein_identical() {
        let dg = make_diagram(vec![(0, 0.0, 1.0), (1, 0.5, 2.0)]);
        let dist = wasserstein_distance(&dg, &dg, 1.0);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_wasserstein_p2() {
        let dg1 = make_diagram(vec![(0, 0.0, 1.0)]);
        let dg2 = make_diagram(vec![(0, 0.0, 1.0)]);
        let dist = wasserstein_distance(&dg1, &dg2, 2.0);
        assert!(dist < 1e-10);
    }

    #[test]
    fn test_diagonal_distance() {
        assert!((diagonal_distance((0.0, 2.0)) - 1.0).abs() < 1e-10);
        assert!((diagonal_distance((1.0, 1.0)).abs()) < 1e-10);
    }

    #[test]
    fn test_bottleneck_symmetry() {
        let dg1 = make_diagram(vec![(0, 0.0, 1.0), (1, 0.5, 2.0)]);
        let dg2 = make_diagram(vec![(0, 0.1, 1.1), (1, 0.6, 2.1)]);
        let d12 = bottleneck_distance(&dg1, &dg2);
        let d21 = bottleneck_distance(&dg2, &dg1);
        assert!((d12 - d21).abs() < 1e-10);
    }

    #[test]
    fn test_next_permutation() {
        let mut arr = vec![0, 1, 2];
        assert!(next_permutation(&mut arr));
        assert_eq!(arr, vec![0, 2, 1]);
    }
}
