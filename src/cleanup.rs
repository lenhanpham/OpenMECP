//! Automated file cleanup for quantum chemistry calculations.
//!
//! This module provides functionality to automatically clean up temporary files
//! generated during quantum chemistry calculations, particularly for ORCA calculations.
//! The cleanup system uses a smart approach to prevent bus errors from excessive
//! temporary files while preserving all essential files.
//!
//! # Philosophy: Smart File Management
//!
//! This implementation uses an intelligent file management strategy:
//! - **Always keep**: Output files (.out, .log) and input files (.in, .inp)
//! - **Latest only**: Energy/gradient files (.engrad) - only from the most recent step
//! - **Configurable**: User-specified extensions from omecp_config.cfg
//! - **Delete everything else**: All temporary and intermediate files
//!
//! This prevents bus errors from accumulating thousands of temporary files
//! during long MECP optimization runs.
//!
//! # Features
//!
//! - **Automatic cleanup** after each QM calculation completes
//! - **Step-based .engrad filtering** - keeps only the latest .engrad files
//! - **Whitelist preservation** - preserves essential output and input files
//! - **Configurable via omecp_config.cfg** for output file extensions
//! - **Program-specific handling** (ORCA, Gaussian, XTB, etc.)
//! - **Comprehensive logging** of all cleanup operations
//! - **Safe operations** with proper error handling
//!
//! # Configuration
//!
//! The cleanup behavior is controlled via omecp_config.cfg. Add the cleanup
//! section to your omecp_config.cfg file:
//!
//! ```ini
//! [cleanup]
//! # Enable or disable automatic cleanup (default: true)
//! enabled = true
//!
//! # Verbose logging for cleanup operations (default: 1)
//! # 0 = quiet, 1 = normal, 2 = verbose
//! verbose = 1
//!
//! # Additional file extensions to preserve (comma-separated)
//! # preserve_extensions = gbw,tmp,backup
//! ```
//!
//! Note: Output file extension is controlled by the `[extensions]` section:
//! ```ini
//! [extensions]
//! orca = out        # All .out files will be preserved
//! ```
//!
//! # File Preservation Strategy
//!
//! ## Files Always Preserved (Never Deleted)
//! - **Output files** (.out, .log, etc.) - Calculation results
//! - **Input files** (.in, .inp) - Input files for calculations
//!
//! ## Files Latest Only (Step-Based Filtering)
//! - **Energy/gradient files** (.engrad) - Only from the most recent optimization step
//!   - Format: `{N}_state_{A|B}.engrad` where N is the step number
//!   - Keeps: Files with maximum step number (e.g., `60_state_A.engrad`)
//!   - Deletes: All other .engrad files (e.g., `59_state_*.engrad`, `58_state_*.engrad`, etc.)
//!
//! ## User-Configurable Extensions
//! - Additional file extensions specified in `omecp_config.cfg` under `[cleanup]` section
//! - All files with these extensions are preserved
//!
//! ## Files Always Deleted
//! - SCF iteration files (.scf)
//! - Temporary files (.tmp, .trash)
//! - Lock files (.lock)
//! - Old .engrad files (older than the latest step)
//! - Any other file types not in the whitelist
//!
//! # Usage Example
//!
//! ```rust
//! use omecp::cleanup::{CleanupManager, CleanupConfig};
//! use omecp::settings::SettingsManager;
//! use omecp::config::QMProgram;
//! use std::path::Path;
//!
//! let settings_manager = SettingsManager::load()?;
//! let program = QMProgram::Orca;
//! let cleanup_config = CleanupConfig::from_settings_manager(&settings_manager, program);
//!
//! let manager = CleanupManager::new(cleanup_config, program);
//! // Clean up files in the job directory
//! manager.cleanup_directory(Path::new("compound_x"))?;
//! ```
//!
//! # Error Handling
//!
//! All cleanup operations return proper `Result` types and log errors
//! without panicking. This ensures that cleanup failures don't interrupt
//! the main calculation workflow.

use crate::config::QMProgram;
use log::{debug, error, info, warn};
use regex::Regex;
use std::fs;
use std::path::Path;
use thiserror::Error;

/// Errors that can occur during cleanup operations.
#[derive(Error, Debug)]
pub enum CleanupError {
    /// I/O error during file operations
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Invalid path error
    #[error("Invalid path: {0}")]
    InvalidPath(String),
}

/// Result type for cleanup operations
pub type Result<T> = std::result::Result<T, CleanupError>;

/// Configuration for cleanup operations.
#[derive(Debug, Clone)]
pub struct CleanupConfig {
    /// Enable automatic cleanup
    pub enabled: bool,

    /// File extensions to preserve (whitelist)
    pub preserve_extensions: Vec<String>,

    /// Verbosity level for cleanup logging
    pub verbose: u32,

    /// Perform cleanup every N optimization steps (default: 5)
    /// Set to 0 to disable periodic cleanup
    pub cleanup_frequency: u32,

    /// Global print level from general settings (0=quiet, 1=normal, 2=verbose)
    pub print_level: u32,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            preserve_extensions: Vec::new(),
            verbose: 1,
            cleanup_frequency: 5,
            print_level: 0,
        }
    }
}

impl CleanupConfig {
    /// Creates a cleanup configuration from settings manager.
    ///
    /// This integrates with omecp_config.cfg to get the user-specified output
    /// extension for the QM program and adds it to the whitelist.
    ///
    /// # Arguments
    ///
    /// * `settings_manager` - Settings manager containing configuration
    /// * `program` - QM program type
    ///
    /// # Returns
    ///
    /// Returns a CleanupConfig with the whitelist of files to preserve
    pub fn from_settings_manager(
        settings_manager: &crate::settings::SettingsManager,
        program: QMProgram,
    ) -> Self {
        let settings = &settings_manager.settings();

        // Get base config from settings
        let mut config = CleanupConfig {
            enabled: settings.cleanup.enabled,
            preserve_extensions: settings.cleanup.preserve_extensions.clone(),
            verbose: settings.cleanup.verbose,
            cleanup_frequency: settings.cleanup.cleanup_frequency,
            print_level: settings.general.print_level,
        };

        // Get user-specified output extension for this program
        let user_ext = settings_manager.get_output_extension(program);

        // Add user-specified output extension to whitelist (always preserve it)
        if !user_ext.is_empty() && !config.preserve_extensions.iter().any(|s| s == user_ext) {
            config.preserve_extensions.push(user_ext.to_string());
        }

        // Add program-specific essential files to whitelist
        // Note: .engrad is handled separately with step-based filtering
        let essential_extensions = match program {
            // Gaussian: only needs output extension
            QMProgram::Gaussian => vec![],
            // ORCA: needs .gbw in addition to output (engrad is step-filtered)
            QMProgram::Orca => vec!["gbw".to_string()],
            // XTB: only needs output extension
            QMProgram::Xtb => vec![],
            // BAGEL: only needs output extension
            QMProgram::Bagel => vec![],
            // Custom: only needs output extension
            QMProgram::Custom => vec![],
        };

        // Add essential extensions to whitelist
        for ext in essential_extensions {
            if !config.preserve_extensions.contains(&ext) {
                config.preserve_extensions.push(ext);
            }
        }

        if config.verbose >= 2 {
            info!("Cleanup configuration for {:?}:", program);
            info!("  Enabled: {}", config.enabled);
            info!("  User output extension: {}", user_ext);
            info!("  Whitelist extensions: {:?}", config.preserve_extensions);
        }

        config
    }

    /// Gets the list of preserve extensions
    pub fn get_preserve_extensions(&self) -> &[String] {
        &self.preserve_extensions
    }

    /// Checks if cleanup is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Gets verbosity level
    pub fn verbosity(&self) -> u32 {
        self.verbose
    }

    /// Gets cleanup frequency (every N steps)
    pub fn cleanup_frequency(&self) -> u32 {
        self.cleanup_frequency
    }

    /// Checks if logging should occur based on print_level and verbose settings.
    ///
    /// This combines the global print_level with the cleanup-specific verbose setting:
    /// - If print_level is 0 (quiet), no cleanup messages are printed regardless of verbose
    /// - If print_level is 1 (normal), messages are printed based on verbose level
    /// - If print_level is 2 (verbose), all messages are printed
    ///
    /// # Arguments
    ///
    /// * `min_verbose_level` - Minimum verbose level required (0, 1, or 2)
    ///
    /// # Returns
    ///
    /// Returns `true` if logging should occur, `false` otherwise
    pub fn should_log(&self, min_verbose_level: u32) -> bool {
        // If global print_level is 0 (quiet), suppress all cleanup output
        if self.print_level == 0 {
            return false;
        }

        // If global print_level is 2 (verbose), allow all messages
        if self.print_level >= 2 {
            return true;
        }

        // If global print_level is 1 (normal), check verbose level
        // verbose = 0: quiet, verbose = 1: normal, verbose = 2: verbose
        self.verbose >= min_verbose_level
    }
}

/// Manages cleanup operations for quantum chemistry calculations.
pub struct CleanupManager {
    /// Cleanup configuration with whitelist
    config: CleanupConfig,

    /// QM program type
    program: QMProgram,
}

impl CleanupManager {
    /// Creates a new cleanup manager.
    ///
    /// # Arguments
    ///
    /// * `config` - Cleanup configuration with whitelist
    /// * `program` - QM program type
    ///
    /// # Returns
    ///
    /// Returns a new CleanupManager instance
    pub fn new(config: CleanupConfig, program: QMProgram) -> Self {
        Self { config, program }
    }

    /// Cleans up temporary files in the specified directory.
    ///
    /// Uses a smart approach: preserves essential files and keeps only the
    /// latest .engrad files to prevent bus errors from excessive files.
    ///
    /// # Arguments
    ///
    /// * `directory` - Path to the directory to clean (e.g., job directory from input file stem)
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success or a `CleanupError` on failure
    pub fn cleanup_directory(&self, directory: &Path) -> Result<()> {
        if !self.config.enabled {
            if self.config.should_log(1) {
                info!(
                    "Cleanup is disabled, skipping directory: {}",
                    directory.display()
                );
            }
            return Ok(());
        }

        if !directory.exists() {
            if self.config.should_log(2) {
                debug!(
                    "Directory does not exist, skipping: {}",
                    directory.display()
                );
            }
            return Ok(());
        }

        if !directory.is_dir() {
            return Err(CleanupError::InvalidPath(format!(
                "Path is not a directory: {}",
                directory.display()
            )));
        }

        if self.config.should_log(1) {
            info!("Starting cleanup in directory: {}", directory.display());
            if self.config.should_log(2) {
                info!(
                    "Preserving files with extensions: {:?}",
                    self.config.preserve_extensions
                );
            }
        }

        // Read all directory entries
        let entries = fs::read_dir(directory).map_err(CleanupError::Io)?;

        let mut all_files = Vec::new();
        for entry in entries {
            match entry {
                Ok(entry) => {
                    let path = entry.path();
                    let filename = path
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();

                    // Skip hidden files and directories
                    if !filename.starts_with('.') && !path.is_dir() {
                        all_files.push((path, filename));
                    }
                }
                Err(e) => {
                    warn!("Error reading directory entry: {}", e);
                }
            }
        }

        // Find the maximum step number from .inp files
        let max_step = self.find_max_step_number(&all_files);

        let mut cleaned_files = Vec::new();
        let mut preserved_files = Vec::new();
        let mut errors = Vec::new();

        for (path, filename) in all_files {
            let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");

            // Check if this file should be preserved
            if self.should_preserve_file(extension, &path, &filename, max_step) {
                preserved_files.push(path.clone());
                if self.config.should_log(2) {
                    debug!("Preserving file: {}", path.display());
                }
            } else {
                // Delete the file
                match fs::remove_file(&path) {
                    Ok(_) => {
                        cleaned_files.push(path.clone());
                        if self.config.should_log(1) {
                            info!("Cleaned up file: {}", path.display());
                        }
                    }
                    Err(e) => {
                        warn!("Failed to remove file {}: {}", path.display(), e);
                        errors.push((path.clone(), e));
                    }
                }
            }
        }

        // Log summary
        if self.config.should_log(1) {
            info!(
                "Cleanup completed: {} files deleted, {} files preserved",
                cleaned_files.len(),
                preserved_files.len()
            );
            if max_step > 0 && self.config.should_log(1) {
                info!("Latest step number: {}", max_step);
            }
        }

        if !errors.is_empty() {
            error!("Cleanup completed with {} errors", errors.len());
        }

        Ok(())
    }

    /// Finds the maximum step number from .inp files in the directory.
    ///
    /// Scans for files matching the pattern `{N}_state_{A|B}.inp` and returns
    /// the maximum step number N found.
    ///
    /// # Arguments
    ///
    /// * `files` - List of (path, filename) tuples
    ///
    /// # Returns
    ///
    /// Returns the maximum step number found, or 0 if no step-based files exist
    fn find_max_step_number(&self, files: &[(std::path::PathBuf, String)]) -> usize {
        let inp_regex = Regex::new(r"^(\d+)_state_[AB]\.inp$").unwrap();
        let mut max_step = 0;

        for (_, filename) in files {
            if let Some(caps) = inp_regex.captures(filename) {
                if let Ok(step) = caps[1].parse::<usize>() {
                    if step > max_step {
                        max_step = step;
                    }
                }
            }
        }

        max_step
    }

    /// Extracts the step number from a .engrad filename.
    ///
    /// Parses files matching the pattern `{N}_state_{A|B}.engrad` and returns
    /// the step number N.
    ///
    /// # Arguments
    ///
    /// * `filename` - The filename to parse
    ///
    /// # Returns
    ///
    /// Returns Some(step_number) if the filename matches the pattern, None otherwise
    fn extract_step_from_engrad(&self, filename: &str) -> Option<usize> {
        let engrad_regex = Regex::new(r"^(\d+)_state_[AB]\.engrad$").unwrap();
        engrad_regex
            .captures(filename)
            .and_then(|caps| caps[1].parse::<usize>().ok())
    }

    /// Determines if a file should be preserved (whitelist check with step-based filtering).
    ///
    /// Files are preserved based on:
    /// 1. Extension in whitelist (always keep)
    /// 2. Special filename patterns (always keep)
    /// 3. .engrad files from the latest step (keep only)
    /// 4. All other files (delete)
    ///
    /// # Arguments
    ///
    /// * `extension` - File extension without the dot
    /// * `path` - Full file path
    /// * `filename` - Just the filename
    /// * `max_step` - Maximum step number from .inp files
    ///
    /// # Returns
    ///
    /// Returns `true` if the file should be preserved, `false` otherwise
    pub fn should_preserve_file(
        &self,
        extension: &str,
        _path: &Path,
        filename: &str,
        max_step: usize,
    ) -> bool {
        // Always preserve essential file types
        if extension == "out" || extension == "log" || extension == "in" || extension == "inp" {
            return true;
        }

        // Preserve input.inp (even without .inp extension check)
        if filename == "input.inp" {
            return true;
        }

        // Special handling for .engrad files - only keep from the latest step
        if extension == "engrad" {
            if let Some(step) = self.extract_step_from_engrad(filename) {
                // Keep only .engrad files from the maximum step
                return step == max_step;
            }
            // .engrad files that don't match the pattern should be deleted
            return false;
        }

        // Whitelist check: preserve if extension is in our list
        if self
            .config
            .preserve_extensions
            .iter()
            .any(|ext| ext == extension)
        {
            return true;
        }

        // Program-specific preservation of special files
        match self.program {
            QMProgram::Orca => {
                // ORCA checkpoint and state files (.gbw) - already in whitelist
            }

            QMProgram::Gaussian => {
                // Gaussian checkpoint files (.chk) - already in whitelist
            }

            _ => {
                // Other programs
            }
        }

        // Not in whitelist, should be deleted
        false
    }

    /// Cleans up a single file if it's not in the whitelist.
    ///
    /// Note: For .engrad files, this method conservatively deletes them unless
    /// they are from the latest step. To use step-based filtering, use
    /// cleanup_directory() instead.
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the file to clean
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if file was deleted, `Ok(false)` if preserved
    pub fn cleanup_file(&self, file_path: &Path) -> Result<bool> {
        if !self.config.enabled {
            return Ok(false);
        }

        if !file_path.exists() || file_path.is_dir() {
            return Ok(false);
        }

        let filename = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");

        let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");

        // For .engrad files, conservatively delete them unless we can determine
        // they should be preserved (use cleanup_directory for step-based filtering)
        if extension == "engrad" {
            // .engrad files are deleted by default (not in preserve_extensions)
            // unless cleanup_directory determines they should be kept
            if self.config.should_log(2) {
                debug!(
                    "Deleting .engrad file (use cleanup_directory for step-based filtering): {}",
                    file_path.display()
                );
            }
            match fs::remove_file(file_path) {
                Ok(_) => {
                    if self.config.should_log(1) {
                        info!("Cleaned up file: {}", file_path.display());
                    }
                    Ok(true)
                }
                Err(e) => {
                    error!("Failed to remove file {}: {}", file_path.display(), e);
                    Err(CleanupError::Io(e))
                }
            }
        } else {
            // For non-.engrad files, use normal whitelist check
            if self.should_preserve_file_simple(extension, filename) {
                if self.config.should_log(2) {
                    debug!("Preserving file: {}", file_path.display());
                }
                return Ok(false);
            }

            match fs::remove_file(file_path) {
                Ok(_) => {
                    if self.config.should_log(1) {
                        info!("Cleaned up file: {}", file_path.display());
                    }
                    Ok(true)
                }
                Err(e) => {
                    error!("Failed to remove file {}: {}", file_path.display(), e);
                    Err(CleanupError::Io(e))
                }
            }
        }
    }

    /// Simplified whitelist check for files that don't need step-based filtering.
    ///
    /// # Arguments
    ///
    /// * `extension` - File extension without the dot
    /// * `filename` - Just the filename
    ///
    /// # Returns
    ///
    /// Returns `true` if the file should be preserved, `false` otherwise
    fn should_preserve_file_simple(&self, extension: &str, filename: &str) -> bool {
        // Always preserve essential file types
        if extension == "out" || extension == "log" || extension == "in" || extension == "inp" {
            return true;
        }

        // Preserve input.inp (even without .inp extension check)
        if filename == "input.inp" {
            return true;
        }

        // Whitelist check: preserve if extension is in our list
        if self
            .config
            .preserve_extensions
            .iter()
            .any(|ext| ext == extension)
        {
            return true;
        }

        // Not in whitelist, should be deleted
        false
    }

    /// Gets the cleanup configuration (read-only).
    pub fn config(&self) -> &CleanupConfig {
        &self.config
    }

    /// Checks if a file would be preserved without actually cleaning it
    ///
    /// Note: For .engrad files, this returns false (use cleanup_directory for step-based filtering)
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to check
    ///
    /// # Returns
    ///
    /// Returns `true` if the file would be preserved (is in whitelist)
    pub fn would_preserve(&self, file_path: &Path) -> bool {
        if !file_path.exists() || file_path.is_dir() {
            return false;
        }

        let filename = file_path.file_name().and_then(|s| s.to_str()).unwrap_or("");

        let extension = file_path.extension().and_then(|s| s.to_str()).unwrap_or("");

        // For .engrad files, conservatively return false
        if extension == "engrad" {
            return false;
        }

        self.should_preserve_file_simple(extension, filename)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::TempDir;

    fn create_test_cleanup_config() -> crate::cleanup::CleanupConfig {
        // Create a test configuration directly without SettingsManager
        let config = crate::cleanup::CleanupConfig {
            enabled: true,
            preserve_extensions: vec!["out".to_string(), "gbw".to_string()],
            verbose: 2,
            cleanup_frequency: 5,
            print_level: 2,
        };
        config
    }

    #[test]
    fn test_preserves_whitelist_files() {
        let config = create_test_cleanup_config();
        let manager = CleanupManager::new(config, QMProgram::Orca);

        // These should be preserved (in whitelist or always keep)
        assert!(manager.should_preserve_file("out", Path::new("test.out"), "test.out", 0));
        assert!(manager.should_preserve_file("log", Path::new("test.log"), "test.log", 0));
        assert!(manager.should_preserve_file("in", Path::new("test.in"), "test.in", 0));
        assert!(manager.should_preserve_file("inp", Path::new("test.inp"), "test.inp", 0));
        assert!(manager.should_preserve_file("gbw", Path::new("test.gbw"), "test.gbw", 0));
        assert!(manager.should_preserve_file("input.inp", Path::new("input.inp"), "input.inp", 0));
    }

    #[test]
    fn test_deletes_non_whitelist_files() {
        let config = create_test_cleanup_config();
        let manager = CleanupManager::new(config, QMProgram::Orca);

        // These should be DELETED (not in whitelist)
        assert!(!manager.should_preserve_file("tmp", Path::new("test.tmp"), "test.tmp", 0));
        assert!(!manager.should_preserve_file("scf", Path::new("test.scf"), "test.scf", 0));
        assert!(!manager.should_preserve_file("bak", Path::new("test.bak"), "test.bak", 0));
        assert!(!manager.should_preserve_file("trash", Path::new("test.trash"), "test.trash", 0));
        assert!(!manager.should_preserve_file("lock", Path::new("test.lock"), "test.lock", 0));
    }

    #[test]
    fn test_engrad_step_based_filtering() {
        let config = create_test_cleanup_config();
        let manager = CleanupManager::new(config, QMProgram::Orca);

        // With max_step = 60
        // These .engrad files should be PRESERVED (from latest step)
        assert!(manager.should_preserve_file(
            "engrad",
            Path::new("60_state_A.engrad"),
            "60_state_A.engrad",
            60
        ));
        assert!(manager.should_preserve_file(
            "engrad",
            Path::new("60_state_B.engrad"),
            "60_state_B.engrad",
            60
        ));

        // These .engrad files should be DELETED (from old steps)
        assert!(!manager.should_preserve_file(
            "engrad",
            Path::new("59_state_A.engrad"),
            "59_state_A.engrad",
            60
        ));
        assert!(!manager.should_preserve_file(
            "engrad",
            Path::new("59_state_B.engrad"),
            "59_state_B.engrad",
            60
        ));
        assert!(!manager.should_preserve_file(
            "engrad",
            Path::new("45_state_A.engrad"),
            "45_state_A.engrad",
            60
        ));
        assert!(!manager.should_preserve_file(
            "engrad",
            Path::new("10_state_B.engrad"),
            "10_state_B.engrad",
            60
        ));

        // .engrad files that don't match the pattern should be DELETED
        assert!(!manager.should_preserve_file(
            "engrad",
            Path::new("test.engrad"),
            "test.engrad",
            60
        ));
        assert!(!manager.should_preserve_file(
            "engrad",
            Path::new("random.engrad"),
            "random.engrad",
            60
        ));
    }

    #[test]
    fn test_cleanup_directory_deletes_non_whitelist() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create test files
        let _ = File::create(dir_path.join("test.out")).unwrap(); // Should be preserved
        let _ = File::create(dir_path.join("test.gbw")).unwrap(); // Should be preserved
        let _ = File::create(dir_path.join("test.tmp")).unwrap(); // Should be deleted
        let _ = File::create(dir_path.join("test.scf")).unwrap(); // Should be deleted
        let _ = File::create(dir_path.join("test.trash")).unwrap(); // Should be deleted

        let config = create_test_cleanup_config();
        let manager = CleanupManager::new(config, QMProgram::Orca);

        // Run cleanup
        let result = manager.cleanup_directory(&dir_path);
        assert!(result.is_ok());

        // Verify cleanup - whitelist files preserved
        assert!(dir_path.join("test.out").exists());
        assert!(dir_path.join("test.gbw").exists());
        // Non-whitelist files deleted
        assert!(!dir_path.join("test.tmp").exists());
        assert!(!dir_path.join("test.scf").exists());
        assert!(!dir_path.join("test.trash").exists());
    }

    #[test]
    fn test_cleanup_directory_step_based_engrad() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create .inp files to determine max step
        let _ = File::create(dir_path.join("45_state_A.inp")).unwrap();
        let _ = File::create(dir_path.join("45_state_B.inp")).unwrap();
        let _ = File::create(dir_path.join("60_state_A.inp")).unwrap();
        let _ = File::create(dir_path.join("60_state_B.inp")).unwrap();

        // Create .engrad files
        let _ = File::create(dir_path.join("45_state_A.engrad")).unwrap();
        let _ = File::create(dir_path.join("45_state_B.engrad")).unwrap();
        let _ = File::create(dir_path.join("60_state_A.engrad")).unwrap();
        let _ = File::create(dir_path.join("60_state_B.engrad")).unwrap();

        let config = create_test_cleanup_config();
        let manager = CleanupManager::new(config, QMProgram::Orca);

        // Run cleanup
        let result = manager.cleanup_directory(&dir_path);
        assert!(result.is_ok());

        // Verify - only 60_state_*.engrad files are preserved
        assert!(dir_path.join("60_state_A.engrad").exists());
        assert!(dir_path.join("60_state_B.engrad").exists());
        assert!(!dir_path.join("45_state_A.engrad").exists());
        assert!(!dir_path.join("45_state_B.engrad").exists());
    }

    #[test]
    fn test_respects_user_output_extension() {
        use std::io::Write;

        // Create a temporary omecp_config.cfg file
        let temp_dir = TempDir::new().unwrap();
        let settings_path = temp_dir.path().join("omecp_config.cfg");

        // Write custom settings with ORCA extension set to "custom"
        let mut file = File::create(&settings_path).unwrap();
        writeln!(file, "[extensions]").unwrap();
        writeln!(file, "orca = custom").unwrap();

        // Change to temp directory and load settings
        let old_cwd = std::env::current_dir().unwrap();
        std::env::set_current_dir(temp_dir.path()).unwrap();

        // Load settings from the file we just created
        let settings_manager = crate::settings::SettingsManager::load().unwrap();
        let config = CleanupConfig::from_settings_manager(&settings_manager, QMProgram::Orca);

        // Restore working directory
        std::env::set_current_dir(old_cwd).unwrap();

        // Should preserve "custom" (user-specified)
        assert!(config.preserve_extensions.contains(&"custom".to_string()));
        // Should also preserve program-specific files (gbw)
        assert!(config.preserve_extensions.contains(&"gbw".to_string()));
        // Should NOT preserve engrad by default (it's step-filtered)
        assert!(!config.preserve_extensions.contains(&"engrad".to_string()));
    }

    #[test]
    fn test_cleanup_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().to_path_buf();

        // Create test files
        let _ = File::create(dir_path.join("test.tmp")).unwrap();

        let config = CleanupConfig {
            enabled: false,
            preserve_extensions: vec!["out".to_string(), "gbw".to_string()],
            verbose: 1,
            cleanup_frequency: 5,
            print_level: 1,
        };

        let manager = CleanupManager::new(config, QMProgram::Orca);

        // Run cleanup (should not delete anything)
        let result = manager.cleanup_directory(&dir_path);
        assert!(result.is_ok());

        // Verify nothing was deleted
        assert!(dir_path.join("test.tmp").exists());
    }
}
