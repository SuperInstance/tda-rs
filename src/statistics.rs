//! Statistical TDA: bootstrap for persistence diagrams, confidence sets.

use crate::persistence::{compute_persistent_homology, PersistenceDiagram};
use crate::distance::bottleneck_distance;
use nalgebra::DVector;
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Result of a bootstrap persistence analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Bootstrap persistence diagrams.
    pub diagrams: Vec<PersistenceDiagram>,
    /// Mean bottleneck distance from bootstrap to original.
    pub mean_bottleneck: f64,
    /// Standard deviation of bottleneck distances.
    pub std_bottleneck: f64,
    /// Confidence band (mean + alpha * std for given confidence level).
    pub confidence_band: f64,
}

/// Perform bootstrap resampling for persistence diagram stability.
pub fn bootstrap_persistence(
    points: &[DVector<f64>],
    n_bootstrap: usize,
    max_dim: usize,
    confidence_level: f64,
) -> BootstrapResult {
    let original_dg = compute_persistent_homology(points, max_dim);
    let mut rng = rand::thread_rng();
    let n = points.len();

    let mut bootstrap_diagrams: Vec<PersistenceDiagram> = Vec::new();
    let mut distances: Vec<f64> = Vec::new();

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let mut sample: Vec<DVector<f64>> = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sample.push(points[idx].clone());
        }

        let dg = compute_persistent_homology(&sample, max_dim);
        let dist = bottleneck_distance(&original_dg, &dg);
        bootstrap_diagrams.push(dg);
        distances.push(dist);
    }

    let mean = if distances.is_empty() { 0.0 } else {
        distances.iter().sum::<f64>() / distances.len() as f64
    };

    let variance = if distances.is_empty() { 0.0 } else {
        distances.iter()
            .map(|d| (d - mean).powi(2))
            .sum::<f64>() / distances.len() as f64
    };
    let std = variance.sqrt();

    // Confidence band using normal approximation
    // For confidence_level (e.g., 0.95), use z-score
    let z = z_score(confidence_level);
    let band = mean + z * std;

    BootstrapResult {
        diagrams: bootstrap_diagrams,
        mean_bottleneck: mean,
        std_bottleneck: std,
        confidence_band: band,
    }
}

/// Approximate z-score for given confidence level.
fn z_score(confidence: f64) -> f64 {
    // Simple approximation
    if confidence >= 0.99 { 2.576 }
    else if confidence >= 0.975 { 2.241 }
    else if confidence >= 0.95 { 1.96 }
    else if confidence >= 0.90 { 1.645 }
    else if confidence >= 0.80 { 1.282 }
    else { 1.0 }
}

/// A confidence set for persistence diagrams.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceSet {
    /// The center persistence diagram.
    pub center: PersistenceDiagram,
    /// The radius (bottleneck distance).
    pub radius: f64,
    /// The confidence level.
    pub level: f64,
}

impl ConfidenceSet {
    /// Check if a persistence diagram falls within this confidence set.
    pub fn contains(&self, diagram: &PersistenceDiagram) -> bool {
        bottleneck_distance(&self.center, diagram) <= self.radius
    }
}

/// Compute a confidence set for a persistence diagram.
pub fn confidence_set(
    points: &[DVector<f64>],
    max_dim: usize,
    confidence_level: f64,
    n_bootstrap: usize,
) -> ConfidenceSet {
    let result = bootstrap_persistence(points, n_bootstrap, max_dim, confidence_level);
    ConfidenceSet {
        center: compute_persistent_homology(points, max_dim),
        radius: result.confidence_band,
        level: confidence_level,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bootstrap_persistence_basic() {
        let points: Vec<DVector<f64>> = (0..4).map(|i| {
            DVector::from_vec(vec![i as f64, 0.0])
        }).collect();
        let result = bootstrap_persistence(&points, 2, 1, 0.95);
        assert_eq!(result.diagrams.len(), 2);
        assert!(result.mean_bottleneck >= 0.0);
        assert!(result.confidence_band >= result.mean_bottleneck);
    }

    #[test]
    fn test_bootstrap_single_point() {
        let points = vec![DVector::from_vec(vec![0.0, 0.0])];
        let result = bootstrap_persistence(&points, 3, 1, 0.95);
        // All bootstraps are the same point
        assert_eq!(result.diagrams.len(), 3);
    }

    #[test]
    fn test_confidence_set() {
        let points: Vec<DVector<f64>> = (0..4).map(|i| {
            DVector::from_vec(vec![i as f64, 0.0])
        }).collect();
        let cs = confidence_set(&points, 1, 0.95, 2);
        assert!(cs.radius >= 0.0);
        assert!((cs.level - 0.95).abs() < 1e-10);
        // The original diagram should be in its own confidence set
        let original_dg = compute_persistent_homology(&points, 1);
        assert!(cs.contains(&original_dg));
    }

    #[test]
    fn test_z_score() {
        assert!((z_score(0.95) - 1.96).abs() < 0.01);
        assert!((z_score(0.99) - 2.576).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_result_serialization() {
        let result = BootstrapResult {
            diagrams: vec![],
            mean_bottleneck: 0.5,
            std_bottleneck: 0.1,
            confidence_band: 0.7,
        };
        let json = serde_json::to_string(&result).unwrap();
        let r2: BootstrapResult = serde_json::from_str(&json).unwrap();
        assert!((r2.mean_bottleneck - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_set_serialization() {
        let cs = ConfidenceSet {
            center: PersistenceDiagram::new(vec![]),
            radius: 1.0,
            level: 0.95,
        };
        let json = serde_json::to_string(&cs).unwrap();
        let cs2: ConfidenceSet = serde_json::from_str(&json).unwrap();
        assert!((cs2.radius - 1.0).abs() < 1e-10);
    }
}
