//! Checkpoint system for saving and restarting MECP calculations.
//!
//! This module provides functionality to save the complete state of a MECP
//! optimization calculation and restart from that state later. This is essential
//! for:
//!
//! - **Long calculations**: Save progress periodically
//! - **Interrupted jobs**: Restart from the last checkpoint
//! - **Parameter testing**: Resume with different settings
//! - **Convergence recovery**: Restart from near-converged geometries
//!
//! # Checkpoint Format
//!
//! Checkpoints are stored as JSON files containing:
//!
//! - **Step number**: Current optimization step
//! - **Geometry**: Current molecular geometry
//! - **Optimization history**: Geometries, gradients, Hessians (for DIIS)
//! - **Hessian matrix**: Current approximate Hessian
//! - **Configuration**: Complete calculation settings
//!
//! # Serialization Strategy
//!
//! The checkpoint system uses serde for serialization. Since some types
//! (DVector, DMatrix) from nalgebra don't directly serialize, wrapper types
//! are used:
//!
//! - `SerializableGeometry`: Converts DVector coords to `Vec<f64>`
//! - `SerializableOptimizationState`: Converts collections of DVector/DMatrix to Vec
//!
//! # Usage
//!
//! Checkpoints are automatically created during optimization but can also be
//! used manually:
//!
//! ```no_run
//! use omecp::checkpoint::Checkpoint;
//! use std::path::Path;
//!
//! // Save checkpoint
//! let checkpoint = Checkpoint::new(step, &geometry, &x_old, &hessian, &opt_state, &config);
//! checkpoint.save(Path::new("mecp.chk"))?;
//!
//! // Load checkpoint
//! let (step, geometry, x_old, hessian, opt_state, config) =
//!     Checkpoint::load(Path::new("mecp.chk"))?;
//! ```
//!
//! # File Location
//!
//! Checkpoint files are typically saved as `mecp.chk` or a user-specified name
//! in the `running_dir/` directory for easy management.

//! Checkpoint system for restarting MECP optimizations.
//!
//! This module provides functionality to save and restore optimization state
//! including geometry, gradients, Hessian, and history for recovery.

use crate::config::Config;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use crate::geometry::Geometry;
use crate::optimizer::OptimizationState;
use nalgebra::{DMatrix, DVector};


// Serializable versions of structs containing nalgebra types

/// Serializable wrapper for Geometry.
///
/// Since `Geometry` contains a `DVector<f64>` which doesn't directly serialize,
/// this wrapper converts the vector to a plain `Vec<f64>` for JSON serialization.
#[derive(Serialize, Deserialize)]
pub struct SerializableGeometry {
    /// Chemical element symbols
    elements: Vec<String>,
    /// Flattened coordinates as `Vec<f64>`
    coords: Vec<f64>,
    /// Number of atoms
    num_atoms: usize,
}

impl From<&Geometry> for SerializableGeometry {
    fn from(geom: &Geometry) -> Self {
        Self {
            elements: geom.elements.clone(),
            coords: geom.coords.data.as_vec().clone(),
            num_atoms: geom.num_atoms,
        }
    }
}

impl From<SerializableGeometry> for Geometry {
    fn from(ser_geom: SerializableGeometry) -> Self {
        Geometry::new(ser_geom.elements, ser_geom.coords)
    }
}

/// Serializable wrapper for OptimizationState.
///
/// Converts collections of nalgebra types (DVector, DMatrix) to plain Vec
/// types for JSON serialization. Preserves the optimization history needed
/// for DIIS convergence acceleration.
#[derive(Serialize, Deserialize)]
pub struct SerializableOptimizationState {
    /// Lagrange multipliers for constraints
    lambdas: Vec<f64>,
    /// Lagrange multiplier for energy difference constraint (FixDE mode)
    lambda_de: Option<f64>,
    /// History of geometries (for DIIS)
    geom_history: Vec<Vec<f64>>,
    /// History of gradients (for DIIS)
    grad_history: Vec<Vec<f64>>,
    /// History of Hessian matrices (for BFGS updates)
    hess_history: Vec<Vec<Vec<f64>>>,
    /// Maximum history size to maintain
    max_history: usize,
}

impl From<&OptimizationState> for SerializableOptimizationState {
    fn from(opt_state: &OptimizationState) -> Self {
        Self {
            lambdas: opt_state.lambdas.clone(),
            lambda_de: opt_state.lambda_de,
            geom_history: opt_state.geom_history.iter().map(|v| v.data.as_vec().clone()).collect(),
            grad_history: opt_state.grad_history.iter().map(|v| v.data.as_vec().clone()).collect(),
            hess_history: opt_state.hess_history.iter().map(|m| {
                m.row_iter().map(|row| row.iter().cloned().collect()).collect()
            }).collect(),
            max_history: opt_state.max_history,
        }
    }
}

impl From<SerializableOptimizationState> for OptimizationState {
    fn from(ser_opt_state: SerializableOptimizationState) -> Self {
        let mut opt_state = OptimizationState::new();
        opt_state.lambdas = ser_opt_state.lambdas;
        opt_state.lambda_de = ser_opt_state.lambda_de;
        opt_state.max_history = ser_opt_state.max_history;

        for coords in ser_opt_state.geom_history {
            opt_state.geom_history.push_back(DVector::from_vec(coords));
        }
        for grad in ser_opt_state.grad_history {
            opt_state.grad_history.push_back(DVector::from_vec(grad));
        }
        for hess in ser_opt_state.hess_history {
            let nrows = hess.len();
            let ncols = hess[0].len();
            let flat: Vec<f64> = hess.into_iter().flatten().collect();
            opt_state.hess_history.push_back(DMatrix::from_row_slice(nrows, ncols, &flat));
        }

        opt_state
    }
}

/// Checkpoint structure for saving/restoring MECP calculations.
///
/// A checkpoint contains all information needed to continue an optimization
/// from a specific step, including the current geometry, optimization history,
/// and calculation configuration.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    /// Current optimization step number
    pub step: usize,
    /// Current molecular geometry
    pub geometry: SerializableGeometry,
    /// Previous geometry (for gradient computation)
    pub x_old: Vec<f64>,
    /// Current approximate Hessian matrix
    pub hessian: Vec<Vec<f64>>,
    /// Optimization state with history
    pub opt_state: SerializableOptimizationState,
    /// Complete calculation configuration
    pub config: Config,
}

/// Loaded checkpoint contents returned by `Checkpoint::load`.
///
/// This struct contains the fully reconstructed runtime types converted from the
/// serialized `Checkpoint` representation. It includes everything required to
/// resume an optimization: the step counter, the molecular geometry, previous
/// coordinates used for gradient differences, the current approximate Hessian,
/// the optimization state (with DIIS/history structures), and the calculation
/// configuration.
pub struct CheckpointLoad {
    /// Optimization step number at the time of saving.
    pub step: usize,
    /// Molecular geometry reconstructed from the checkpoint.
    pub geometry: Geometry,
    /// Previous geometry coordinates (flattened) used for finite differences.
    pub x_old: DVector<f64>,
    /// Hessian matrix (approximate) reconstructed as a nalgebra `DMatrix`.
    pub hessian: DMatrix<f64>,
    /// Optimization state including history, lambdas, and other runtime data.
    pub opt_state: OptimizationState,
    /// Calculation configuration used when the checkpoint was created.
    pub config: Config,
}

impl Checkpoint {
    /// Create a new checkpoint from current optimization state.
    ///
    /// # Arguments
    ///
    /// * `step` - Current optimization step number
    /// * `geometry` - Current molecular geometry
    /// * `x_old` - Previous geometry coordinates
    /// * `hessian` - Current approximate Hessian matrix
    /// * `opt_state` - Optimization state with history
    /// * `config` - Calculation configuration
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::checkpoint::Checkpoint;
    /// use omecp::geometry::Geometry;
    /// use omecp::optimizer::OptimizationState;
    /// use omecp::config::Config;
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// let elements = vec!["H".to_string()];
    /// let coords = vec![0.0, 0.0, 0.0];
    /// let geometry = Geometry::new(elements, coords);
    /// let x_old = vec![0.0, 0.0, 0.0];
    /// let hessian = DMatrix::identity(3, 3);
    /// let opt_state = OptimizationState::new();
    /// let config = Config::default();
    ///
    /// let checkpoint = Checkpoint::new(
    ///     5,                          // Step 5
    ///     &geometry,                  // Current geometry
    ///     &DVector::from_vec(x_old),  // Previous coords
    ///     &hessian,                   // Hessian matrix
    ///     &opt_state,                 // Optimization state
    ///     &config,                    // Configuration
    /// );
    /// ```
    pub fn new(
        step: usize,
        geometry: &Geometry,
        x_old: &DVector<f64>,
        hessian: &DMatrix<f64>,
        opt_state: &OptimizationState,
        config: &Config,
    ) -> Self {
        Self {
            step,
            geometry: geometry.into(),
            x_old: x_old.data.as_vec().clone(),
            hessian: hessian.row_iter().map(|row| row.iter().cloned().collect()).collect(),
            opt_state: opt_state.into(),
            config: config.clone(),
        }
    }

    /// Save checkpoint to a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where checkpoint will be saved
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be written
    /// - Serialization fails
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::checkpoint::Checkpoint;
    /// use std::path::Path;
    /// use omecp::geometry::Geometry;
    /// use omecp::optimizer::OptimizationState;
    /// use omecp::config::Config;
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let elements = vec!["H".to_string()];
    ///     let coords = vec![0.0, 0.0, 0.0];
    ///     let geometry = Geometry::new(elements, coords);
    ///     let x_old = vec![0.0, 0.0, 0.0];
    ///     let hessian = DMatrix::identity(3, 3);
    ///     let opt_state = OptimizationState::new();
    ///     let config = Config::default();
    ///
    ///     let checkpoint = Checkpoint::new(
    ///         5,
    ///         &geometry,
    ///         &DVector::from_vec(x_old),
    ///         &hessian,
    ///         &opt_state,
    ///         &config,
    ///     );
    ///
    ///     checkpoint.save(Path::new("mecp.chk"))?;
    ///     std::fs::remove_file("mecp.chk")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn save(&self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load checkpoint from a JSON file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to checkpoint file
    ///
    /// # Returns
    ///
    /// Returns a `CheckpointLoad` struct containing:
    /// - `step`: Optimization step number
    /// - `geometry`: Molecular geometry
    /// - `x_old`: Previous geometry coordinates
    /// - `hessian`: Hessian matrix
    /// - `opt_state`: Optimization state
    /// - `config`: Calculation configuration
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - Deserialization fails
    /// - File format is invalid
    ///
    /// # Examples
    ///
    /// ```
    /// use omecp::checkpoint::{Checkpoint, CheckpointLoad};
    /// use std::path::Path;
    /// use omecp::geometry::Geometry;
    /// use omecp::optimizer::OptimizationState;
    /// use omecp::config::Config;
    /// use nalgebra::{DMatrix, DVector};
    ///
    /// fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let elements = vec!["H".to_string()];
    ///     let coords = vec![0.0, 0.0, 0.0];
    ///     let geometry = Geometry::new(elements, coords);
    ///     let x_old = vec![0.0, 0.0, 0.0];
    ///     let hessian = DMatrix::identity(3, 3);
    ///     let opt_state = OptimizationState::new();
    ///     let config = Config::default();
    ///
    ///     let checkpoint = Checkpoint::new(
    ///         5,
    ///         &geometry,
    ///         &DVector::from_vec(x_old),
    ///         &hessian,
    ///         &opt_state,
    ///         &config,
    ///     );
    ///
    ///     checkpoint.save(Path::new("mecp.chk"))?;
    ///
    ///     let loaded: CheckpointLoad = Checkpoint::load(Path::new("mecp.chk"))?;
    ///
    ///     std::fs::remove_file("mecp.chk")?;
    ///     Ok(())
    /// }
    /// ```
    pub fn load(path: &Path) -> Result<CheckpointLoad, Box<dyn std::error::Error>> {
        let content = fs::read_to_string(path)?;
        let checkpoint: Checkpoint = serde_json::from_str(&content)?;

        let geometry = Geometry::from(checkpoint.geometry);
        let x_old = DVector::from_vec(checkpoint.x_old);
        let nrows = checkpoint.hessian.len();
        let ncols = if nrows > 0 { checkpoint.hessian[0].len() } else { 0 };
        let hess_flat: Vec<f64> = checkpoint.hessian.into_iter().flatten().collect();
        let hessian = if nrows > 0 && ncols > 0 {
            DMatrix::from_row_slice(nrows, ncols, &hess_flat)
        } else {
            DMatrix::from_row_slice(0, 0, &[])
        };
        let opt_state = OptimizationState::from(checkpoint.opt_state);

        Ok(CheckpointLoad {
            step: checkpoint.step,
            geometry,
            x_old,
            hessian,
            opt_state,
            config: checkpoint.config,
        })
    }
}
