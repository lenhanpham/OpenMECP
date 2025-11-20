//! Dynamic file naming based on input file basename
//!
//! This module provides a centralized system for generating file names dynamically
//! based on the input file basename. This allows multiple jobs to run in the same
//! directory without file conflicts.
//!
//! # Example
//!
//! ```
//! use std::path::Path;
//! use omecp::naming::FileNaming;
//!
//! let input_path = Path::new("compound_xyz_123.input");
//! let naming = FileNaming::new(input_path);
//!
//! // Generate dynamic file names
//! assert_eq!(naming.state_a_chk(), "compound_xyz_123_state_A.chk");
//! assert_eq!(naming.pre_a("job_dir", "inp"), "job_dir/compound_xyz_123_pre_A.inp");
//! assert_eq!(naming.step_state_a("job_dir", 5, "gjf"), "job_dir/compound_xyz_123_5_state_A.gjf");
//! ```

use std::path::Path;

/// Manages dynamic file naming based on input file basename
///
/// All file names are prefixed with the basename extracted from the input file,
/// ensuring unique file names when multiple jobs run in the same directory.
#[derive(Debug, Clone)]
pub struct FileNaming {
    basename: String,
}

impl FileNaming {
    /// Creates a new FileNaming instance from an input file path
    ///
    /// Extracts the file stem (filename without extension) to use as the basename
    /// for all generated file names.
    ///
    /// # Arguments
    ///
    /// * `input_path` - Path to the input file
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use omecp::naming::FileNaming;
    ///
    /// let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
    /// assert_eq!(naming.basename(), "compound_xyz_123");
    /// ```
    pub fn new(input_path: &Path) -> Self {
        let basename = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mecp_job")
            .to_string();

        Self { basename }
    }

    /// Returns the basename used for file naming
    pub fn basename(&self) -> &str {
        &self.basename
    }

    // Checkpoint files (Gaussian)

    /// Returns the state A checkpoint file name
    ///
    /// Format: `{basename}_state_A.chk`
    pub fn state_a_chk(&self) -> String {
        format!("{}_state_A.chk", self.basename)
    }

    /// Returns the state B checkpoint file name
    ///
    /// Format: `{basename}_state_B.chk`
    pub fn state_b_chk(&self) -> String {
        format!("{}_state_B.chk", self.basename)
    }

    /// Returns the 'a' checkpoint file name
    ///
    /// Format: `{basename}_a.chk`
    pub fn a_chk(&self) -> String {
        format!("{}_a.chk", self.basename)
    }

    /// Returns the 'b' checkpoint file name
    ///
    /// Format: `{basename}_b.chk`
    pub fn b_chk(&self) -> String {
        format!("{}_b.chk", self.basename)
    }

    /// Returns the state A checkpoint file path in a job directory
    ///
    /// Format: `{job_dir}/{basename}_state_A.chk`
    pub fn state_a_chk_path(&self, job_dir: &str) -> String {
        format!("{}/{}", job_dir, self.state_a_chk())
    }

    /// Returns the state B checkpoint file path in a job directory
    ///
    /// Format: `{job_dir}/{basename}_state_B.chk`
    pub fn state_b_chk_path(&self, job_dir: &str) -> String {
        format!("{}/{}", job_dir, self.state_b_chk())
    }

    /// Returns the 'a' checkpoint file path in a job directory
    ///
    /// Format: `{job_dir}/{basename}_a.chk`
    pub fn a_chk_path(&self, job_dir: &str) -> String {
        format!("{}/{}", job_dir, self.a_chk())
    }

    /// Returns the 'b' checkpoint file path in a job directory
    ///
    /// Format: `{job_dir}/{basename}_b.chk`
    pub fn b_chk_path(&self, job_dir: &str) -> String {
        format!("{}/{}", job_dir, self.b_chk())
    }

    // Wavefunction files (ORCA)

    /// Returns the state A wavefunction file path
    ///
    /// Format: `{job_dir}/{basename}_state_A.gbw`
    pub fn state_a_gbw(&self, job_dir: &str) -> String {
        format!("{}/{}_state_A.gbw", job_dir, self.basename)
    }

    /// Returns the state B wavefunction file path
    ///
    /// Format: `{job_dir}/{basename}_state_B.gbw`
    pub fn state_b_gbw(&self, job_dir: &str) -> String {
        format!("{}/{}_state_B.gbw", job_dir, self.basename)
    }

    /// Returns the 'a' wavefunction file path
    ///
    /// Format: `{job_dir}/{basename}_a.gbw`
    pub fn a_gbw(&self, job_dir: &str) -> String {
        format!("{}/{}_a.gbw", job_dir, self.basename)
    }

    /// Returns the 'b' wavefunction file path
    ///
    /// Format: `{job_dir}/{basename}_b.gbw`
    pub fn b_gbw(&self, job_dir: &str) -> String {
        format!("{}/{}_b.gbw", job_dir, self.basename)
    }

    // Pre-point calculation files

    /// Returns the pre-point A input file path
    ///
    /// Format: `{job_dir}/{basename}_pre_A.{ext}`
    pub fn pre_a(&self, job_dir: &str, ext: &str) -> String {
        format!("{}/{}_pre_A.{}", job_dir, self.basename, ext)
    }

    /// Returns the pre-point B input file path
    ///
    /// Format: `{job_dir}/{basename}_pre_B.{ext}`
    pub fn pre_b(&self, job_dir: &str, ext: &str) -> String {
        format!("{}/{}_pre_B.{}", job_dir, self.basename, ext)
    }

    /// Returns the pre-point A checkpoint file path
    ///
    /// Format: `{job_dir}/{basename}_pre_A.chk`
    pub fn pre_a_chk(&self, job_dir: &str) -> String {
        format!("{}/{}_pre_A.chk", job_dir, self.basename)
    }

    /// Returns the pre-point B checkpoint file path
    ///
    /// Format: `{job_dir}/{basename}_pre_B.chk`
    pub fn pre_b_chk(&self, job_dir: &str) -> String {
        format!("{}/{}_pre_B.chk", job_dir, self.basename)
    }

    /// Returns the pre-point A wavefunction file path
    ///
    /// Format: `{job_dir}/{basename}_pre_A.gbw`
    pub fn pre_a_gbw(&self, job_dir: &str) -> String {
        format!("{}/{}_pre_A.gbw", job_dir, self.basename)
    }

    /// Returns the pre-point B wavefunction file path
    ///
    /// Format: `{job_dir}/{basename}_pre_B.gbw`
    pub fn pre_b_gbw(&self, job_dir: &str) -> String {
        format!("{}/{}_pre_B.gbw", job_dir, self.basename)
    }

    // Optimization step files

    /// Returns the state A input file path for a given step
    ///
    /// Format: `{job_dir}/{basename}_{step}_state_A.{ext}`
    pub fn step_state_a(&self, job_dir: &str, step: usize, ext: &str) -> String {
        format!("{}/{}_{}_state_A.{}", job_dir, self.basename, step, ext)
    }

    /// Returns the state B input file path for a given step
    ///
    /// Format: `{job_dir}/{basename}_{step}_state_B.{ext}`
    pub fn step_state_b(&self, job_dir: &str, step: usize, ext: &str) -> String {
        format!("{}/{}_{}_state_B.{}", job_dir, self.basename, step, ext)
    }

    /// Returns the state A wavefunction file path for a given step
    ///
    /// Format: `{job_dir}/{basename}_{step}_state_A.gbw`
    pub fn step_state_a_gbw(&self, job_dir: &str, step: usize) -> String {
        format!("{}/{}_{}_state_A.gbw", job_dir, self.basename, step)
    }

    /// Returns the state B wavefunction file path for a given step
    ///
    /// Format: `{job_dir}/{basename}_{step}_state_B.gbw`
    pub fn step_state_b_gbw(&self, job_dir: &str, step: usize) -> String {
        format!("{}/{}_{}_state_B.gbw", job_dir, self.basename, step)
    }

    /// Returns the state A engrad file path for a given step
    ///
    /// Format: `{job_dir}/{basename}_{step}_state_A.engrad`
    pub fn step_state_a_engrad(&self, job_dir: &str, step: usize) -> String {
        format!("{}/{}_{}_state_A.engrad", job_dir, self.basename, step)
    }

    /// Returns the state B engrad file path for a given step
    ///
    /// Format: `{job_dir}/{basename}_{step}_state_B.engrad`
    pub fn step_state_b_engrad(&self, job_dir: &str, step: usize) -> String {
        format!("{}/{}_{}_state_B.engrad", job_dir, self.basename, step)
    }

    // Alternative naming for PES analysis modes

    /// Returns the 'A' input file path for a given step (PES analysis)
    ///
    /// Format: `{job_dir}/{basename}_{step}_A.{ext}`
    pub fn step_a(&self, job_dir: &str, step: usize, ext: &str) -> String {
        format!("{}/{}_{}_A.{}", job_dir, self.basename, step, ext)
    }

    /// Returns the 'B' input file path for a given step (PES analysis)
    ///
    /// Format: `{job_dir}/{basename}_{step}_B.{ext}`
    pub fn step_b(&self, job_dir: &str, step: usize, ext: &str) -> String {
        format!("{}/{}_{}_B.{}", job_dir, self.basename, step, ext)
    }

    // Special files for specific modes

    /// Returns a drive calculation file path
    ///
    /// Format: `{job_dir}/{basename}_drive_{step}_{state}.{ext}`
    pub fn drive_file(&self, job_dir: &str, step: usize, state: &str, ext: &str) -> String {
        format!(
            "{}/{}_drive_{}_{}.{}",
            job_dir, self.basename, step, state, ext
        )
    }

    /// Returns a NEB calculation file path
    ///
    /// Format: `{job_dir}/{basename}_neb_{step}_{state}.{ext}`
    pub fn neb_file(&self, job_dir: &str, step: usize, state: &str, ext: &str) -> String {
        format!(
            "{}/{}_neb_{}_{}.{}",
            job_dir, self.basename, step, state, ext
        )
    }

    /// Returns the basename for ORCA gbw file references in input headers
    ///
    /// This is used in ORCA input files where the gbw path needs to be specified.
    /// Format: `{job_dir}/{basename}`
    pub fn orca_basename(&self, job_dir: &str) -> String {
        format!("{}/{}", job_dir, self.basename)
    }

    // Final output files

    /// Returns the final MECP geometry output file name
    ///
    /// Format: `{basename}_mecp.xyz`
    ///
    /// # Example
    ///
    /// ```
    /// use std::path::Path;
    /// use omecp::naming::FileNaming;
    ///
    /// let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
    /// assert_eq!(naming.final_mecp_xyz(), "compound_xyz_123_mecp.xyz");
    /// ```
    pub fn final_mecp_xyz(&self) -> String {
        format!("{}_mecp.xyz", self.basename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_final_mecp_xyz() {
        let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
        assert_eq!(naming.final_mecp_xyz(), "compound_xyz_123_mecp.xyz");
    }

    #[test]
    fn test_final_mecp_xyz_with_different_extensions() {
        let naming1 = FileNaming::new(Path::new("test.inp"));
        assert_eq!(naming1.final_mecp_xyz(), "test_mecp.xyz");

        let naming2 = FileNaming::new(Path::new("molecule.gjf"));
        assert_eq!(naming2.final_mecp_xyz(), "molecule_mecp.xyz");

        let naming3 = FileNaming::new(Path::new("calc_001.input"));
        assert_eq!(naming3.final_mecp_xyz(), "calc_001_mecp.xyz");
    }

    #[test]
    fn test_final_mecp_xyz_with_path() {
        let naming = FileNaming::new(Path::new("/path/to/compound_xyz_123.input"));
        assert_eq!(naming.final_mecp_xyz(), "compound_xyz_123_mecp.xyz");
    }

    #[test]
    fn test_basename_extraction() {
        let naming = FileNaming::new(Path::new("compound_xyz_123.input"));
        assert_eq!(naming.basename(), "compound_xyz_123");
    }
}
