//! Configuration management for OpenMECP.
//!
//! This module provides a flexible configuration system that allows users to customize
//! program behavior through INI-format configuration files. The system supports
//! hierarchical configuration with the following precedence:
//!
//! 1. Local configuration (`./settings.ini`)
//! 2. User configuration (`~/.config/omecp/settings.ini`)
//! 3. System configuration (`/etc/omecp/settings.ini`)
//! 4. Built-in defaults
//!
//! # Configuration File Format
//!
//! The configuration uses INI format with sections for different types of settings:
//!
//! ```ini
//! [extensions]
//! gaussian = log
//! orca = out
//! xtb = out
//! bagel = json
//! custom = log
//!
//! [general]
//! max_memory = 4GB
//! default_nprocs = 4
//! temp_directory = /tmp/omecp
//!
//! [logging]
//! level = info
//! file = omecp.log
//! ```
//!
//! # Usage
//!
//! ```rust
//! use omecp::settings::SettingsManager;
//! use omecp::config::QMProgram;
//!
//! let settings = SettingsManager::load()?;
//! let extension = settings.get_output_extension(QMProgram::Orca);
//! println!("ORCA output extension: {}", extension);
//! ```

use crate::config::QMProgram;
use configparser::ini::Ini;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that can occur during configuration loading and processing.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// I/O error when reading configuration files
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    /// INI parsing error
    #[error("INI parsing error: {0}")]
    IniParse(String),
    /// Invalid configuration value
    #[error("Invalid configuration value: {0}")]
    InvalidValue(String),
    /// Missing required configuration section
    #[error("Missing required section: {0}")]
    MissingSection(String),
}

/// Main configuration structure containing all program settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// File extension settings for different QM programs
    pub extensions: ExtensionSettings,
    /// General program settings
    pub general: GeneralSettings,
    /// Logging configuration
    pub logging: LoggingSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            extensions: ExtensionSettings::default(),
            general: GeneralSettings::default(),
            logging: LoggingSettings::default(),
        }
    }
}

/// File extension settings for different quantum chemistry programs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionSettings {
    /// Gaussian output file extension (default: "log")
    pub gaussian: String,
    /// ORCA output file extension (default: "out")
    pub orca: String,
    /// XTB output file extension (default: "out")
    pub xtb: String,
    /// BAGEL output file extension (default: "json")
    pub bagel: String,
    /// Custom QM program output file extension (default: "log")
    pub custom: String,
}

impl Default for ExtensionSettings {
    fn default() -> Self {
        Self {
            gaussian: "log".to_string(),
            orca: "out".to_string(),
            xtb: "out".to_string(),
            bagel: "json".to_string(),
            custom: "log".to_string(),
        }
    }
}

/// General program settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralSettings {
    /// Maximum memory usage (default: "4GB")
    pub max_memory: String,
    /// Default number of processors (default: 4)
    pub default_nprocs: u32,
    /// Temporary directory for calculations (default: "/tmp/omecp")
    pub temp_directory: String,
    /// Print level for file operations and verbose output (default: 0)
    /// 0 = quiet (minimal output), 1 = normal, 2 = verbose (show all file operations)
    pub print_level: u32,
}

impl Default for GeneralSettings {
    fn default() -> Self {
        Self {
            max_memory: "4GB".to_string(),
            default_nprocs: 4,
            temp_directory: "/tmp/omecp".to_string(),
            print_level: 0, // Default to quiet mode
        }
    }
}

/// Logging configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingSettings {
    /// Log level (default: "info")
    pub level: String,
    /// Log file path (default: "omecp.log")
    pub file: String,
}

impl Default for LoggingSettings {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file: "omecp.log".to_string(),
        }
    }
}

/// Configuration manager that handles loading and accessing program settings.
pub struct SettingsManager {
    settings: Settings,
    config_source: String,
}

impl SettingsManager {
    /// Loads configuration from available configuration files.
    ///
    /// Searches for configuration files in the following order:
    /// 1. `./settings.ini` (current working directory)
    /// 2. `~/.config/omecp/settings.ini` (user configuration)
    /// 3. `/etc/omecp/settings.ini` (system configuration)
    /// 4. Built-in defaults (fallback)
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing:
    /// - `Ok(SettingsManager)` - Successfully loaded configuration
    /// - `Err(ConfigError)` - Configuration loading failed
    ///
    /// # Examples
    ///
    /// ```rust
    /// use omecp::settings::SettingsManager;
    ///
    /// let settings = SettingsManager::load()?;
    /// println!("Configuration loaded from: {}", settings.config_source());
    /// ```
    pub fn load() -> Result<Self, ConfigError> {
        let (settings, source) = Self::load_from_files()?;
        info!("Configuration loaded from: {}", source);
        Ok(Self {
            settings,
            config_source: source,
        })
    }

    /// Returns the source of the loaded configuration.
    pub fn config_source(&self) -> &str {
        &self.config_source
    }

    /// Gets the output file extension for the specified QM program.
    ///
    /// # Arguments
    ///
    /// * `program` - The quantum chemistry program
    ///
    /// # Returns
    ///
    /// The file extension (without the dot) as a string slice
    ///
    /// # Examples
    ///
    /// ```rust
    /// use omecp::config::QMProgram;
    /// use omecp::settings::SettingsManager;
    ///
    /// let settings = SettingsManager::load()?;
    /// let extension = settings.get_output_extension(QMProgram::Orca);
    /// assert_eq!(extension, "out");
    /// ```
    pub fn get_output_extension(&self, program: QMProgram) -> &str {
        match program {
            QMProgram::Gaussian => &self.settings.extensions.gaussian,
            QMProgram::Orca => &self.settings.extensions.orca,
            QMProgram::Xtb => &self.settings.extensions.xtb,
            QMProgram::Bagel => &self.settings.extensions.bagel,
            QMProgram::Custom => &self.settings.extensions.custom,
        }
    }

    /// Gets the general settings.
    pub fn general(&self) -> &GeneralSettings {
        &self.settings.general
    }

    /// Gets the logging settings.
    pub fn logging(&self) -> &LoggingSettings {
        &self.settings.logging
    }

    /// Gets the extension settings.
    pub fn extensions(&self) -> &ExtensionSettings {
        &self.settings.extensions
    }

    /// Loads configuration from files with hierarchical precedence.
    fn load_from_files() -> Result<(Settings, String), ConfigError> {
        let mut settings = Settings::default();
        let mut config_source = "built-in defaults".to_string();

        // Try to load system configuration
        if let Some(system_path) = Self::get_system_config_path() {
            if system_path.exists() {
                match Self::load_config(&system_path) {
                    Ok(system_config) => {
                        settings.merge(system_config);
                        config_source = format!("system config ({})", system_path.display());
                        debug!("Loaded system configuration from: {}", system_path.display());
                    }
                    Err(e) => {
                        warn!("Failed to load system config from {}: {}", system_path.display(), e);
                    }
                }
            }
        }

        // Try to load user configuration (overrides system)
        if let Some(user_path) = Self::get_user_config_path() {
            if user_path.exists() {
                match Self::load_config(&user_path) {
                    Ok(user_config) => {
                        settings.merge(user_config);
                        config_source = format!("user config ({})", user_path.display());
                        debug!("Loaded user configuration from: {}", user_path.display());
                    }
                    Err(e) => {
                        warn!("Failed to load user config from {}: {}", user_path.display(), e);
                    }
                }
            }
        }

        // Try to load local configuration (overrides user)
        let local_path = PathBuf::from("settings.ini");
        if local_path.exists() {
            match Self::load_config(&local_path) {
                Ok(local_config) => {
                    settings.merge(local_config);
                    config_source = format!("local config ({})", local_path.display());
                    debug!("Loaded local configuration from: {}", local_path.display());
                }
                Err(e) => {
                    warn!("Failed to load local config from {}: {}", local_path.display(), e);
                }
            }
        }

        Ok((settings, config_source))
    }

    /// Loads configuration from a single INI file.
    fn load_config(path: &Path) -> Result<Settings, ConfigError> {
        let content = fs::read_to_string(path)?;
        let mut ini = Ini::new();
        ini.read(content)
            .map_err(|e| ConfigError::IniParse(format!("Failed to parse INI: {}", e)))?;

        let mut settings = Settings::default();

        // Load extensions section
        if let Some(extensions_map) = ini.get_map_ref().get("extensions") {
            settings.extensions = Self::parse_extensions(extensions_map)?;
        }

        // Load general section
        if let Some(general_map) = ini.get_map_ref().get("general") {
            settings.general = Self::parse_general(general_map)?;
        }

        // Load logging section
        if let Some(logging_map) = ini.get_map_ref().get("logging") {
            settings.logging = Self::parse_logging(logging_map)?;
        }

        Ok(settings)
    }

    /// Parses the extensions section from INI configuration.
    fn parse_extensions(section: &std::collections::HashMap<String, Option<String>>) -> Result<ExtensionSettings, ConfigError> {
        let mut extensions = ExtensionSettings::default();

        if let Some(Some(gaussian)) = section.get("gaussian") {
            extensions.gaussian = gaussian.clone();
        }
        if let Some(Some(orca)) = section.get("orca") {
            extensions.orca = orca.clone();
        }
        if let Some(Some(xtb)) = section.get("xtb") {
            extensions.xtb = xtb.clone();
        }
        if let Some(Some(bagel)) = section.get("bagel") {
            extensions.bagel = bagel.clone();
        }
        if let Some(Some(custom)) = section.get("custom") {
            extensions.custom = custom.clone();
        }

        Ok(extensions)
    }

    /// Parses the general section from INI configuration.
    fn parse_general(section: &std::collections::HashMap<String, Option<String>>) -> Result<GeneralSettings, ConfigError> {
        let mut general = GeneralSettings::default();

        if let Some(Some(max_memory)) = section.get("max_memory") {
            general.max_memory = max_memory.clone();
        }
        if let Some(Some(default_nprocs)) = section.get("default_nprocs") {
            general.default_nprocs = default_nprocs.parse()
                .map_err(|_| ConfigError::InvalidValue(format!("Invalid default_nprocs: {}", default_nprocs)))?;
        }
        if let Some(Some(temp_directory)) = section.get("temp_directory") {
            general.temp_directory = temp_directory.clone();
        }
        if let Some(Some(print_level)) = section.get("print_level") {
            general.print_level = print_level.parse()
                .map_err(|_| ConfigError::InvalidValue(format!("Invalid print_level: {}", print_level)))?;
        }

        Ok(general)
    }

    /// Parses the logging section from INI configuration.
    fn parse_logging(section: &std::collections::HashMap<String, Option<String>>) -> Result<LoggingSettings, ConfigError> {
        let mut logging = LoggingSettings::default();

        if let Some(Some(level)) = section.get("level") {
            logging.level = level.clone();
        }
        if let Some(Some(file)) = section.get("file") {
            logging.file = file.clone();
        }

        Ok(logging)
    }

    /// Gets the system configuration file path.
    fn get_system_config_path() -> Option<PathBuf> {
        #[cfg(unix)]
        {
            Some(PathBuf::from("/etc/omecp/settings.ini"))
        }
        #[cfg(windows)]
        {
            // On Windows, use ProgramData directory
            std::env::var("PROGRAMDATA")
                .ok()
                .map(|pd| PathBuf::from(pd).join("omecp").join("settings.ini"))
        }
    }

    /// Gets the user configuration file path.
    fn get_user_config_path() -> Option<PathBuf> {
        #[cfg(unix)]
        {
            std::env::var("HOME")
                .ok()
                .map(|home| PathBuf::from(home).join(".config").join("omecp").join("settings.ini"))
        }
        #[cfg(windows)]
        {
            std::env::var("APPDATA")
                .ok()
                .map(|appdata| PathBuf::from(appdata).join("omecp").join("settings.ini"))
        }
    }
}

impl SettingsManager {
    /// Creates a default settings.ini file with all available configuration options.
    ///
    /// This function generates a comprehensive configuration file template with:
    /// - All available configuration sections and parameters
    /// - Default values for each parameter
    /// - Detailed comments explaining each option
    /// - Examples of common customizations
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the settings.ini file should be created
    ///
    /// # Returns
    ///
    /// Returns a `Result` indicating success or failure of file creation
    ///
    /// # Examples
    ///
    /// ```rust
    /// use omecp::settings::SettingsManager;
    /// use std::path::Path;
    ///
    /// SettingsManager::create_template(Path::new("settings.ini"))?;
    /// ```
    pub fn create_template(path: &Path) -> Result<(), ConfigError> {
        let template_content = Self::generate_template_content();
        fs::write(path, template_content)?;
        info!("Created settings template at: {}", path.display());
        Ok(())
    }

    /// Generates the content for a settings.ini template file.
    fn generate_template_content() -> String {
        format!(
r#"# OpenMECP Configuration File
# 
# This file allows you to customize OpenMECP behavior without modifying source code.
# Configuration files are loaded in hierarchical order with local settings taking precedence:
#
# 1. Current working directory (./settings.ini) - highest priority
# 2. User config directory (~/.config/omecp/settings.ini on Unix, %APPDATA%/omecp/settings.ini on Windows)
# 3. System config directory (/etc/omecp/settings.ini on Unix, %PROGRAMDATA%/omecp/settings.ini on Windows)
# 4. Built-in defaults (fallback)
#
# Any missing sections or values will use the built-in defaults shown below.

[extensions]
# Output file extensions for different quantum chemistry programs
# These extensions are used when reading calculation output files

# Gaussian output files (default: log)
gaussian = {}

# ORCA output files (default: out)  
orca = {}

# XTB output files (default: out)
xtb = {}

# BAGEL output files (default: json)
bagel = {}

# Custom QM program output files (default: log)
custom = {}

[general]
# General program settings

# Maximum memory usage (default: 4GB)
# Examples: 1GB, 2GB, 4GB, 8GB, 16GB, 32GB
max_memory = {}

# Default number of processors (default: 4)
# Should match your system's CPU core count for optimal performance
default_nprocs = {}

# Temporary directory for calculations (default: /tmp/omecp)
# Make sure this directory exists and has sufficient space
temp_directory = {}

# Print level for file operations and verbose output (default: 0)
# 0 = quiet (minimal output, recommended for clean logs)
# 1 = normal (standard output with key information)
# 2 = verbose (show all file operations and detailed progress)
print_level = {}

[logging]
# Logging configuration

# Log level: debug, info, warn, error (default: info)
# - debug: Detailed debugging information
# - info: General information about program execution
# - warn: Warning messages about potential issues
# - error: Only error messages
level = {}

# Log file path (default: omecp.log)
# Use absolute path or relative to working directory
file = {}

# Example custom configurations:
#
# For a system with non-standard ORCA setup:
# [extensions]
# orca = out
#
# For high-memory calculations:
# [general]
# max_memory = 32GB
# default_nprocs = 16
#
# For debugging calculations:
# [logging]
# level = debug
# file = debug.log
#
# For custom XTB output format:
# [extensions]
# xtb = xyz
#
# For BAGEL with different output format:
# [extensions]
# bagel = out
"#,
            // Extension defaults
            ExtensionSettings::default().gaussian,
            ExtensionSettings::default().orca,
            ExtensionSettings::default().xtb,
            ExtensionSettings::default().bagel,
            ExtensionSettings::default().custom,
            // General defaults
            GeneralSettings::default().max_memory,
            GeneralSettings::default().default_nprocs,
            GeneralSettings::default().temp_directory,
            GeneralSettings::default().print_level,
            // Logging defaults
            LoggingSettings::default().level,
            LoggingSettings::default().file,
        )
    }
}

impl Settings {
    /// Merges another Settings instance into this one, overriding existing values.
    fn merge(&mut self, other: Settings) {
        // Merge extensions
        if !other.extensions.gaussian.is_empty() {
            self.extensions.gaussian = other.extensions.gaussian;
        }
        if !other.extensions.orca.is_empty() {
            self.extensions.orca = other.extensions.orca;
        }
        if !other.extensions.xtb.is_empty() {
            self.extensions.xtb = other.extensions.xtb;
        }
        if !other.extensions.bagel.is_empty() {
            self.extensions.bagel = other.extensions.bagel;
        }
        if !other.extensions.custom.is_empty() {
            self.extensions.custom = other.extensions.custom;
        }

        // Merge general settings
        if !other.general.max_memory.is_empty() {
            self.general.max_memory = other.general.max_memory;
        }
        if other.general.default_nprocs > 0 {
            self.general.default_nprocs = other.general.default_nprocs;
        }
        if !other.general.temp_directory.is_empty() {
            self.general.temp_directory = other.general.temp_directory;
        }
        if other.general.print_level > 0 {
            self.general.print_level = other.general.print_level;
        }

        // Merge logging settings
        if !other.logging.level.is_empty() {
            self.logging.level = other.logging.level;
        }
        if !other.logging.file.is_empty() {
            self.logging.file = other.logging.file;
        }
    }
}