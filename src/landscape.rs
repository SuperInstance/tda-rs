//! Persistence landscapes: construction, integration, and distances.

use crate::persistence::PersistenceDiagram;
use serde::{Serialize, Deserialize};

/// A persistence landscape is a sequence of piecewise-linear functions λ_k(t).
/// We represent it as a collection of critical points for each lambda.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceLandscape {
    /// Each layer is a sorted list of (t, λ_k(t)) critical points.
    layers: Vec<Vec<(f64, f64)>>,
}

impl PersistenceLandscape {
    /// Construct from a persistence diagram.
    pub fn from_diagram(dg: &PersistenceDiagram) -> Self {
        let mut tent_functions: Vec<Vec<(f64, f64)>> = Vec::new();

        for pair in &dg.pairs {
            if pair.is_essential() {
                continue;
            }
            let birth = pair.birth;
            let death = pair.death;
            let midpoint = (birth + death) / 2.0;
            let height = (death - birth) / 2.0;
            // Tent function: rises from (birth, 0) to (midpoint, height) to (death, 0)
            tent_functions.push(vec![
                (birth, 0.0),
                (midpoint, height),
                (death, 0.0),
            ]);
        }

        if tent_functions.is_empty() {
            return PersistenceLandscape { layers: vec![] };
        }

        // Sort tent functions by their midpoint
        tent_functions.sort_by(|a, b| {
            let ma = (a[0].0 + a[2].0) / 2.0;
            let mb = (b[0].0 + b[2].0) / 2.0;
            ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build layers by taking k-th largest tent function at each point
        let max_layers = tent_functions.len();
        let mut layers: Vec<Vec<(f64, f64)>> = Vec::new();

        // Collect all critical points
        let mut all_t: Vec<f64> = Vec::new();
        for tf in &tent_functions {
            for &(t, _) in tf {
                all_t.push(t);
            }
        }
        all_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        all_t.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        for k in 0..max_layers {
            let mut layer_points: Vec<(f64, f64)> = Vec::new();
            for &t in &all_t {
                // Evaluate all tent functions at t and take (k+1)-th largest
                let mut values: Vec<f64> = tent_functions.iter()
                    .map(|tf| evaluate_tent(tf, t))
                    .filter(|&v| v > 0.0)
                    .collect();
                values.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let v = if k < values.len() { values[k] } else { 0.0 };
                layer_points.push((t, v));
            }

            if layer_points.is_empty() {
                break;
            }

            // Simplify: remove consecutive points with same value
            let mut simplified: Vec<(f64, f64)> = vec![layer_points[0]];
            for i in 1..layer_points.len() {
                let prev = simplified.last().unwrap();
                let cur = layer_points[i];
                // Keep if value changed or significant t gap
                if (prev.1 - cur.1).abs() > 1e-12 || (cur.0 - prev.0).abs() > 1e-12 {
                    simplified.push(cur);
                }
            }
            layers.push(simplified);
        }

        PersistenceLandscape { layers }
    }

    /// Number of landscape layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Evaluate the k-th landscape function at time t.
    pub fn evaluate(&self, k: usize, t: f64) -> f64 {
        if k >= self.layers.len() {
            return 0.0;
        }
        evaluate_piecewise_linear(&self.layers[k], t)
    }

    /// Integrate the k-th landscape function.
    pub fn integrate(&self, k: usize) -> f64 {
        if k >= self.layers.len() {
            return 0.0;
        }
        integrate_piecewise_linear(&self.layers[k])
    }

    /// L^p norm of the landscape.
    pub fn norm_p(&self, p: f64) -> f64 {
        let mut total = 0.0;
        for k in 0..self.layers.len() {
            total += self.layer_norm_p(k, p).powf(p);
        }
        total.powf(1.0 / p)
    }

    fn layer_norm_p(&self, k: usize, p: f64) -> f64 {
        if k >= self.layers.len() {
            return 0.0;
        }
        let pts = &self.layers[k];
        if pts.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        let n = 100; // integration points per segment
        for i in 0..pts.len() - 1 {
            let dt = (pts[i + 1].0 - pts[i].0) / n as f64;
            for j in 0..n {
                let t = pts[i].0 + dt * (j as f64 + 0.5);
                let v = evaluate_piecewise_linear(pts, t);
                total += v.powf(p) * dt;
            }
        }
        total
    }
}

/// Evaluate a tent function at time t.
fn evaluate_tent(tf: &[(f64, f64)], t: f64) -> f64 {
    if tf.len() < 3 { return 0.0; }
    let (b, _) = tf[0];
    let (m, h) = tf[1];
    let (d, _) = tf[2];
    if t < b || t > d { return 0.0; }
    if t <= m {
        h * (t - b) / (m - b)
    } else {
        h * (d - t) / (d - m)
    }
}

/// Evaluate a piecewise-linear function at time t.
fn evaluate_piecewise_linear(pts: &[(f64, f64)], t: f64) -> f64 {
    if pts.is_empty() { return 0.0; }
    if pts.len() == 1 { return if (pts[0].0 - t).abs() < 1e-12 { pts[0].1 } else { 0.0 }; }

    if t <= pts[0].0 { return 0.0; }
    if t >= pts.last().unwrap().0 { return 0.0; }

    for i in 0..pts.len() - 1 {
        if t >= pts[i].0 && t <= pts[i + 1].0 {
            let dt = pts[i + 1].0 - pts[i].0;
            if dt.abs() < 1e-15 { return pts[i].1; }
            let frac = (t - pts[i].0) / dt;
            return pts[i].1 + frac * (pts[i + 1].1 - pts[i].1);
        }
    }
    0.0
}

/// Integrate a piecewise-linear function (trapezoidal).
fn integrate_piecewise_linear(pts: &[(f64, f64)]) -> f64 {
    if pts.len() < 2 { return 0.0; }
    let mut total = 0.0;
    for i in 0..pts.len() - 1 {
        let dt = pts[i + 1].0 - pts[i].0;
        total += dt * (pts[i].1 + pts[i + 1].1) / 2.0;
    }
    total
}

/// L^p distance between two persistence landscapes.
pub fn landscape_distance(l1: &PersistenceLandscape, l2: &PersistenceLandscape, p: f64) -> f64 {
    let max_layers = l1.num_layers().max(l2.num_layers());

    // Collect evaluation points from both landscapes
    let mut all_t: Vec<f64> = Vec::new();
    for k in 0..l1.num_layers() {
        for &(t, _) in &l1.layers[k] {
            all_t.push(t);
        }
    }
    for k in 0..l2.num_layers() {
        for &(t, _) in &l2.layers[k] {
            all_t.push(t);
        }
    }
    all_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_t.dedup();

    if all_t.is_empty() { return 0.0; }

    let mut total = 0.0;
    let n = 100;
    let t_min = all_t.first().unwrap() - 1.0;
    let t_max = all_t.last().unwrap() + 1.0;
    let dt = (t_max - t_min) / n as f64;

    for k in 0..max_layers {
        for i in 0..n {
            let t = t_min + dt * (i as f64 + 0.5);
            let v1 = l1.evaluate(k, t);
            let v2 = l2.evaluate(k, t);
            total += (v1 - v2).abs().powf(p) * dt;
        }
    }

    total.powf(1.0 / p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::PersistencePair;

    fn make_diagram(pairs: Vec<(f64, f64)>) -> PersistenceDiagram {
        PersistenceDiagram::new(
            pairs.into_iter().map(|(b, d)| PersistencePair::new(0, b, d)).collect()
        )
    }

    #[test]
    fn test_landscape_construction() {
        let dg = make_diagram(vec![(0.0, 2.0)]);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        assert!(landscape.num_layers() >= 1);
    }

    #[test]
    fn test_landscape_evaluate_peak() {
        let dg = make_diagram(vec![(0.0, 2.0)]);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        // Peak should be at t=1.0 with value 1.0
        let val = landscape.evaluate(0, 1.0);
        assert!((val - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_landscape_evaluate_outside() {
        let dg = make_diagram(vec![(0.0, 2.0)]);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        let val = landscape.evaluate(0, 5.0);
        assert!(val < 0.01);
    }

    #[test]
    fn test_landscape_integrate() {
        let pairs = vec![PersistencePair::new(1, 0.0, 4.0)];
        let dg = PersistenceDiagram::new(pairs);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        assert!(landscape.num_layers() > 0, "expected at least 1 layer");
        // Check the layer has points
        let layer0 = &landscape.layers[0];
        assert!(layer0.len() >= 2, "layer0 has {} points", layer0.len());
        let integ = landscape.integrate(0);
        assert!(integ > 0.0, "integrate returned {}", integ);
    }

    #[test]
    fn test_landscape_empty_diagram() {
        let dg = PersistenceDiagram::new(vec![]);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        assert_eq!(landscape.num_layers(), 0);
    }

    #[test]
    fn test_landscape_multiple_features() {
        let dg = make_diagram(vec![(0.0, 2.0), (1.0, 3.0)]);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        assert!(landscape.num_layers() >= 1);
    }

    #[test]
    fn test_landscape_distance_identical() {
        let dg = make_diagram(vec![(0.0, 2.0)]);
        let l1 = PersistenceLandscape::from_diagram(&dg);
        let l2 = PersistenceLandscape::from_diagram(&dg);
        let dist = landscape_distance(&l1, &l2, 2.0);
        assert!(dist < 0.5);
    }

    #[test]
    fn test_evaluate_tent() {
        let tf = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        assert!((evaluate_tent(&tf, 0.5) - 0.5).abs() < 1e-10);
        assert!((evaluate_tent(&tf, 1.0) - 1.0).abs() < 1e-10);
        assert!((evaluate_tent(&tf, 1.5) - 0.5).abs() < 1e-10);
        assert!(evaluate_tent(&tf, 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_piecewise_linear() {
        let pts = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 0.0)];
        let area = integrate_piecewise_linear(&pts);
        assert!((area - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_landscape_serialization() {
        let dg = make_diagram(vec![(0.0, 2.0)]);
        let landscape = PersistenceLandscape::from_diagram(&dg);
        let json = serde_json::to_string(&landscape).unwrap();
        let l2: PersistenceLandscape = serde_json::from_str(&json).unwrap();
        assert_eq!(l2.num_layers(), landscape.num_layers());
    }
}
