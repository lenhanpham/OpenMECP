use nalgebra::DMatrix;
use omecp::*;
use omecp::{checkpoint, lst};
use std::env;
use std::path::Path;
use std::process;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    // Check for help flags before processing commands
    check_help_flags(&args);

    let command = &args[1];

    match command.as_str() {
        "ci" => {
            // Create Input template command
            if args.len() < 3 {
                eprintln!("Error: Missing geometry file argument");
                print_usage(&args[0]);
                process::exit(1);
            }
            let geometry_path = Path::new(&args[2]);
            let output_path = args.get(3).map(Path::new);

            match run_create_input(geometry_path, output_path) {
                Ok(output_file) => {
                    println!("✓ Template input file created successfully!");
                    println!("  Output file: {}", output_file.display());
                    println!("\nNext steps:");
                    println!("  1. Review and edit the generated template file");
                    println!("  2. Update parameters (method, basis set, etc.) as needed");
                    println!("  3. Run OpenMECP: {} {}", args[0], output_file.display());
                }
                Err(e) => {
                    eprintln!("Error creating template: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            // Default OpenMECP optimization command
            if !command.starts_with('-') {
                let input_path = Path::new(&args[1]);
                match run_mecp(input_path) {
                    Ok(()) => println!("\n****Congrats! MECP has converged****"),
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        process::exit(1);
                    }
                }
            } else {
                eprintln!("Error: Unknown command: {}", command);
                print_usage(&args[0]);
                process::exit(1);
            }
        }
    }
}

/// Check for help flags and print appropriate help
fn check_help_flags(args: &[String]) {
    use omecp::help::*;

    // Check for global help flags
    if args.len() >= 3 && (args[1] == "--help" || args[1] == "-h") {
        let topic = &args[2];

        match topic.as_str() {
            "keywords" => {
                print_keyword_help();
                process::exit(0);
            }
            "methods" => {
                print_method_help();
                process::exit(0);
            }
            "features" => {
                print_feature_help();
                process::exit(0);
            }
            "examples" => {
                print_examples();
                process::exit(0);
            }
            _ => {
                print_global_help();
                process::exit(0);
            }
        }
    }

    // Check for help without topic (omecp --help or omecp -h)
    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        print_global_help();
        process::exit(0);
    }

    // Check for command-specific help (omecp ci --help)
    if args.len() >= 3 && args[1] == "ci" && (args[2] == "--help" || args[2] == "-h") {
        print_ci_help();
        process::exit(0);
    }

    // Check for help in MECP mode (omecp input.inp --help)
    if args.len() >= 3 && !args[1].starts_with('-') && (args[2] == "--help" || args[2] == "-h") {
        print_mecp_help(&args[1]);
        process::exit(0);
    }
}

/// Print help for MECP optimization mode
fn print_mecp_help(input_file: &str) {
    println!("MECP Optimization - Input File Help");
    println!("═══════════════════════════════════════");
    println!();
    println!("Input file: {}", input_file);
    println!();
    println!("This file contains all parameters and settings for the MECP optimization.");
    println!("For complete documentation, use:");
    println!();
    println!("  omecp --help keywords   # All input file keywords");
    println!("  omecp --help methods    # QM methods and programs");
    println!("  omecp --help features   # MECP features and modes");
    println!("  omecp --help examples   # Usage examples");
    println!();
}

fn print_usage(program_name: &str) {
    eprintln!("OpenMECP - Minimum Energy Crossing Point optimization");
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  {} ci <geometry_file> [output_file]", program_name);
    eprintln!("                    Create a template input file from a geometry file");
    eprintln!();
    eprintln!("  {} <input_file>", program_name);
    eprintln!("                    Run MECP optimization using the input file");
    eprintln!();
    eprintln!("Supported geometry formats:");
    eprintln!("  .xyz  - XYZ coordinate file");
    eprintln!("  .log  - Gaussian output file");
    eprintln!("  .gjf  - Gaussian input file");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} ci molecule.xyz", program_name);
    eprintln!("  {} ci molecule.xyz custom.inp", program_name);
    eprintln!("  {} my_calculation.inp", program_name);
}

fn run_create_input<P: AsRef<Path>>(
    geometry_file: P,
    output_path: Option<P>,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>>
where
    P: AsRef<Path>,
{
    use omecp::template_generator::*;

    let geometry_file = geometry_file.as_ref();
    let geometry_path = geometry_file.canonicalize()?;

    // Validate file exists and has supported extension
    if !geometry_path.exists() {
        return Err(format!("Geometry file not found: {}", geometry_path.display()).into());
    }

    if !is_supported_format(&geometry_path) {
        return Err(format!("Unsupported file format. Supported formats: .xyz, .log, .gjf").into());
    }

    // Extract geometry to validate the file
    println!("Reading geometry from: {}", geometry_path.display());
    let template_content = generate_template_from_file(&geometry_path)?;

    // Determine output path
    let output_path = match output_path {
        Some(p) => p.as_ref().to_path_buf(),
        None => get_default_output_path(&geometry_path),
    };

    // Write template to file
    write_template_to_file(&template_content, &output_path)?;

    Ok(output_path)
}

fn run_mecp(input_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    println!("**** MECP in Rust****");
    println!("****By Le Nhan Pham****\n");

    // Parse input
    let input_data = parser::parse_input(input_path)?;

    println!("Parsed {} atoms", input_data.geometry.num_atoms);
    println!(
        "Program: {:?}, Mode: {:?}",
        input_data.config.program, input_data.config.run_mode
    );

    // Create QM interface
    let qm: Box<dyn qm_interface::QMInterface> = match input_data.config.program {
        config::QMProgram::Gaussian => {
            let command = input_data
                .config
                .program_commands
                .get("gaussian")
                .cloned()
                .unwrap_or_else(|| "g16".to_string());
            Box::new(qm_interface::GaussianInterface::new(
                command,
                input_data.config.mp2,
            ))
        }
        config::QMProgram::Orca => {
            let command = input_data
                .config
                .program_commands
                .get("orca")
                .cloned()
                .unwrap_or_else(|| "orca".to_string());
            Box::new(qm_interface::OrcaInterface::new(command))
        }
        config::QMProgram::Bagel => {
            let command = input_data
                .config
                .program_commands
                .get("bagel")
                .cloned()
                .unwrap_or_else(|| "bagel".to_string());
            // Read BAGEL model template
            let model_path = &input_data.config.bagel_model;
            if model_path.is_empty() {
                return Err("BAGEL requires bagel_model parameter".into());
            }
            let model_template = std::fs::read_to_string(model_path)
                .map_err(|e| format!("Failed to read BAGEL model: {}", e))?;
            Box::new(qm_interface::BagelInterface::new(command, model_template))
        }
        config::QMProgram::Xtb => {
            let command = input_data
                .config
                .program_commands
                .get("xtb")
                .cloned()
                .unwrap_or_else(|| "xtb".to_string());
            Box::new(qm_interface::XtbInterface::new(command))
        }
        config::QMProgram::Custom => {
            if input_data.config.custom_interface_file.is_empty() {
                return Err("Custom QM program requires custom_interface_file parameter".into());
            }
            let interface_path = Path::new(&input_data.config.custom_interface_file);
            Box::new(qm_interface::CustomInterface::from_file(interface_path)?)
        }
    };

    // Create running_dir directory
    std::fs::create_dir_all("running_dir")?;

    // Check if restart is requested
    if input_data.config.restart {
        return run_restart(input_data, &*qm);
    }

    // Check if LST interpolation is requested
    if input_data.lst1.is_some() && input_data.lst2.is_some() {
        return run_lst_interpolation(input_data, &*qm);
    }

    // Check if PES scan is requested
    if !input_data.config.scans.is_empty() {
        return run_pes_scan(input_data, &*qm);
    }

    // Check if coordinate driving is requested
    if input_data.config.run_mode == config::RunMode::CoordinateDrive {
        return run_coordinate_driving(input_data, &*qm);
    }

    // Check if path optimization is requested
    if input_data.config.run_mode == config::RunMode::PathOptimization {
        return run_path_optimization(input_data, &*qm);
    }

    // Check if fix-dE optimization is requested
    if input_data.config.run_mode == config::RunMode::FixDE {
        return run_fix_de(input_data, &*qm);
    }

    // Build headers
    let header_a = io::build_gaussian_header(
        &input_data.config,
        input_data.config.charge1,
        input_data.config.mult1,
        &input_data.config.td1,
    );

    let header_b = io::build_gaussian_header(
        &input_data.config,
        input_data.config.charge2,
        input_data.config.mult2,
        &input_data.config.td2,
    );

    // Run pre-point calculations for stable/inter_read modes
    if input_data.config.run_mode == config::RunMode::Stable
        || input_data.config.run_mode == config::RunMode::InterRead
    {
        run_pre_point(
            &input_data.geometry,
            &header_a,
            &header_b,
            &input_data,
            &*qm,
            input_data.config.run_mode,
        )?;
    }

    let config = input_data.config;
    let mut geometry = input_data.geometry;
    let constraints = &input_data.constraints;
    let fixed_atoms = &input_data.fixed_atoms;

    // Run initial calculations
    println!("\n****Running initial calculations****");
    qm.write_input(
        &geometry,
        &header_a,
        &input_data.tail1,
        Path::new("running_dir/0_A.gjf"),
    )?;
    qm.write_input(
        &geometry,
        &header_b,
        &input_data.tail2,
        Path::new("running_dir/0_B.gjf"),
    )?;

    qm.run_calculation(Path::new("running_dir/0_A.gjf"))?;
    qm.run_calculation(Path::new("running_dir/0_B.gjf"))?;

    let state1 = qm.read_output(Path::new("running_dir/0_A.log"), config.state1)?;
    let state2 = qm.read_output(Path::new("running_dir/0_B.log"), config.state2)?;

    // Initialize optimization
    let mut opt_state = optimizer::OptimizationState::new();
    let mut x_old = geometry.coords.clone();
    let mut hessian = DMatrix::identity(geometry.coords.len(), geometry.coords.len());

    // Main optimization loop
    for step in 0..config.max_steps {
        println!("\n****Step {}****", step + 1);

        // Compute MECP gradient
        let grad = optimizer::compute_mecp_gradient(
            &state1,
            &state2,
            constraints,
            &mut opt_state,
            fixed_atoms,
        );

        // Choose optimizer: BFGS for first 3 steps, then GDIIS/GEDIIS
        let x_new = if step < 3 || !opt_state.has_enough_history() {
            println!("Using BFGS optimizer");
            optimizer::bfgs_step(&x_old, &grad, &hessian, &config)
        } else if config.use_gediis {
            println!("Using GEDIIS optimizer");
            optimizer::gediis_step(&opt_state, &config)
        } else {
            println!("Using GDIIS optimizer");
            optimizer::gdiis_step(&opt_state, &config)
        };

        // Update geometry
        geometry.coords = x_new.clone();

        // Run calculations in parallel
        let step_name_a = format!("running_dir/{}_A.gjf", step + 1);
        let step_name_b = format!("running_dir/{}_B.gjf", step + 1);

        qm.write_input(
            &geometry,
            &header_a,
            &input_data.tail1,
            Path::new(&step_name_a),
        )?;
        qm.write_input(
            &geometry,
            &header_b,
            &input_data.tail2,
            Path::new(&step_name_b),
        )?;

        // Run calculations sequentially
        qm.run_calculation(Path::new(&step_name_a))?;
        qm.run_calculation(Path::new(&step_name_b))?;

        let state1_new = qm.read_output(
            Path::new(&format!("running_dir/{}_A.log", step + 1)),
            config.state1,
        )?;
        let state2_new = qm.read_output(
            Path::new(&format!("running_dir/{}_B.log", step + 1)),
            config.state2,
        )?;

        // Compute new gradient for Hessian update
        let grad_new = optimizer::compute_mecp_gradient(
            &state1_new,
            &state2_new,
            constraints,
            &mut opt_state,
            fixed_atoms,
        );

        // Check convergence
        let conv = optimizer::check_convergence(
            state1_new.energy,
            state2_new.energy,
            &x_old,
            &x_new,
            &grad_new,
            &config,
        );

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state1_new.energy,
            state2_new.energy,
            (state1_new.energy - state2_new.energy).abs()
        );

        if conv.is_converged() {
            println!("\nConverged at step {}", step + 1);
            io::write_xyz(&geometry, Path::new("final.xyz"))?;
            return Ok(());
        }

        // Update Hessian
        let sk = &x_new - &x_old;
        let yk = &grad_new - &grad;
        hessian = optimizer::update_hessian_psb(&hessian, &sk, &yk);

        // Add to history for GDIIS/GEDIIS
        let energy_diff = state1_new.energy - state2_new.energy;
        opt_state.add_to_history(
            x_new.clone(),
            grad_new.clone(),
            hessian.clone(),
            energy_diff,
        );

        // Save checkpoint
        let checkpoint =
            checkpoint::Checkpoint::new(step, &geometry, &x_new, &hessian, &opt_state, &config);
        checkpoint.save(Path::new(&config.checkpoint_file))?;

        x_old = x_new;
    }

    Err("Maximum steps exceeded".into())
}

fn run_pes_scan(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    use config::ScanType;

    println!("\n****Running PES Scan****");

    let config = &input_data.config;
    let mut geometry = input_data.geometry;
    let mut constraints = input_data.constraints.clone();
    let scan1 = &config.scans[0];
    let scan2 = config.scans.get(1);

    let values1: Vec<f64> = (0..scan1.num_points)
        .map(|i| scan1.start + i as f64 * scan1.step_size)
        .collect();

    let values2: Vec<f64> = scan2
        .map(|s| {
            (0..s.num_points)
                .map(|i| s.start + i as f64 * s.step_size)
                .collect()
        })
        .unwrap_or_else(|| vec![0.0]);

    let tail1 = input_data.tail1.clone();
    let tail2 = input_data.tail2.clone();
    let fixed_atoms = input_data.fixed_atoms.clone();

    for &val1 in &values1 {
        let constraint1 = match &scan1.scan_type {
            ScanType::Bond { atoms } => constraints::Constraint::Bond {
                atoms: *atoms,
                target: val1,
            },
            ScanType::Angle { atoms } => constraints::Constraint::Angle {
                atoms: *atoms,
                target: val1.to_radians(),
            },
        };
        constraints.push(constraint1);

        for &val2 in &values2 {
            if let Some(scan2) = scan2 {
                let constraint2 = match &scan2.scan_type {
                    ScanType::Bond { atoms } => constraints::Constraint::Bond {
                        atoms: *atoms,
                        target: val2,
                    },
                    ScanType::Angle { atoms } => constraints::Constraint::Angle {
                        atoms: *atoms,
                        target: val2.to_radians(),
                    },
                };
                constraints.push(constraint2);
            }

            println!("\n****Scan point: {:.4} {:.4}****", val1, val2);

            // Run optimization with constraints
            run_single_optimization(
                &config,
                &mut geometry,
                &constraints,
                &tail1,
                &tail2,
                &fixed_atoms,
                qm,
            )?;

            // Save results
            let filename = format!("scan_{}_{}.xyz", val1, val2);
            io::write_xyz(&geometry, Path::new(&filename))?;

            if scan2.is_some() {
                constraints.pop();
            }
        }
        constraints.pop();
    }

    Ok(())
}

fn run_single_optimization(
    config: &config::Config,
    geometry: &mut geometry::Geometry,
    constraints: &[constraints::Constraint],
    tail1: &str,
    tail2: &str,
    fixed_atoms: &[usize],
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    let header_a = io::build_gaussian_header(config, config.charge1, config.mult1, &config.td1);
    let header_b = io::build_gaussian_header(config, config.charge2, config.mult2, &config.td2);

    qm.write_input(geometry, &header_a, tail1, Path::new("running_dir/0_A.gjf"))?;
    qm.write_input(geometry, &header_b, tail2, Path::new("running_dir/0_B.gjf"))?;
    qm.run_calculation(Path::new("running_dir/0_A.gjf"))?;
    qm.run_calculation(Path::new("running_dir/0_B.gjf"))?;

    let mut opt_state = optimizer::OptimizationState::new();
    let mut x_old = geometry.coords.clone();
    let mut hessian = DMatrix::identity(geometry.coords.len(), geometry.coords.len());

    for step in 0..config.max_steps {
        let state1 = qm.read_output(
            Path::new(&format!("running_dir/{}_A.log", step)),
            config.state1,
        )?;
        let state2 = qm.read_output(
            Path::new(&format!("running_dir/{}_B.log", step)),
            config.state2,
        )?;

        let grad = optimizer::compute_mecp_gradient(
            &state1,
            &state2,
            constraints,
            &mut opt_state,
            fixed_atoms,
        );

        let x_new = if step < 3 || !opt_state.has_enough_history() {
            optimizer::bfgs_step(&x_old, &grad, &hessian, config)
        } else {
            optimizer::gdiis_step(&opt_state, config)
        };

        geometry.coords = x_new.clone();

        qm.write_input(
            geometry,
            &header_a,
            tail1,
            Path::new(&format!("running_dir/{}_A.gjf", step + 1)),
        )?;
        qm.write_input(
            geometry,
            &header_b,
            tail2,
            Path::new(&format!("running_dir/{}_B.gjf", step + 1)),
        )?;
        qm.run_calculation(Path::new(&format!("running_dir/{}_A.gjf", step + 1)))?;
        qm.run_calculation(Path::new(&format!("running_dir/{}_B.gjf", step + 1)))?;

        let state1_new = qm.read_output(
            Path::new(&format!("running_dir/{}_A.log", step + 1)),
            config.state1,
        )?;
        let state2_new = qm.read_output(
            Path::new(&format!("running_dir/{}_B.log", step + 1)),
            config.state2,
        )?;
        let grad_new = optimizer::compute_mecp_gradient(
            &state1_new,
            &state2_new,
            constraints,
            &mut opt_state,
            fixed_atoms,
        );

        let conv = optimizer::check_convergence(
            state1_new.energy,
            state2_new.energy,
            &x_old,
            &x_new,
            &grad_new,
            config,
        );

        if conv.is_converged() {
            return Ok(());
        }

        let sk = &x_new - &x_old;
        let yk = &grad_new - &grad;
        hessian = optimizer::update_hessian_psb(&hessian, &sk, &yk);
        let energy_diff = state1_new.energy - state2_new.energy;
        opt_state.add_to_history(
            x_new.clone(),
            grad_new.clone(),
            hessian.clone(),
            energy_diff,
        );
        x_old = x_new;
    }

    Ok(())
}

fn run_lst_interpolation(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Advanced LST Interpolation****");

    let geom1 = input_data.lst1.as_ref().unwrap();
    let geom2 = input_data.lst2.as_ref().unwrap();
    let config = &input_data.config;

    // Choose interpolation method
    println!("Available LST methods:");
    println!("1. Linear Synchronous Transit (LST) - Simple linear interpolation");
    println!("2. Quadratic Synchronous Transit (QST) - Smooth quadratic interpolation");
    println!("Select method (1-2) [1]: ");

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let method_choice = input.trim().parse::<usize>().unwrap_or(1);

    let method = match method_choice {
        2 => lst::InterpolationMethod::Quadratic,
        _ => lst::InterpolationMethod::Linear,
    };

    // Get number of interpolation points
    println!("Number of interpolation points [10]: ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;
    let num_points = input.trim().parse::<usize>().unwrap_or(10);

    println!(
        "Using {:?} interpolation with {} points",
        method, num_points
    );

    // Generate geometries
    let geometries = lst::interpolate(geom1, geom2, num_points, method);

    // Validate geometries
    match lst::validate_geometries(&geometries) {
        Ok(_) => println!("✓ All geometries validated successfully"),
        Err(e) => {
            println!("✗ Geometry validation failed: {}", e);
            println!("Do you want to continue anyway? (y/n) [n]: ");
            input.clear();
            std::io::stdin().read_line(&mut input)?;
            if input.trim().to_lowercase() != "y" {
                return Ok(());
            }
        }
    }

    // Show geometry preview
    lst::print_geometry_preview(&geometries);

    // Build headers
    let header_a = io::build_gaussian_header(config, config.charge1, config.mult1, &config.td1);
    let header_b = io::build_gaussian_header(config, config.charge2, config.mult2, &config.td2);

    // Write input files
    for (i, geom) in geometries.iter().enumerate() {
        let num = i + 1;
        qm.write_input(
            geom,
            &header_a,
            &input_data.tail1,
            Path::new(&format!("running_dir/{}_A.gjf", num)),
        )?;
        qm.write_input(
            geom,
            &header_b,
            &input_data.tail2,
            Path::new(&format!("running_dir/{}_B.gjf", num)),
        )?;
    }

    println!(
        "\nGenerated {} input files in running_dir/",
        geometries.len() * 2
    );

    // Enhanced confirmation
    println!("\n****Confirmation****");
    println!("Interpolation method: {:?}", method);
    println!("Number of points: {}", num_points);
    println!("Total input files: {}", geometries.len() * 2);
    println!("QM program: {:?}", config.program);
    println!("Method: {}", config.method);

    println!("\nDo you want to run the calculations? (y/n) [n]: ");
    input.clear();
    std::io::stdin().read_line(&mut input)?;

    if input.trim().to_lowercase() != "y" {
        println!("LST interpolation completed. Input files are ready in running_dir/ directory.");
        println!("You can run them manually or modify the geometries as needed.");
        return Ok(());
    }

    // Run calculations and collect energies
    let mut energies_a = Vec::new();
    let mut energies_b = Vec::new();
    let mut successful_points = 0;

    for (i, _) in geometries.iter().enumerate() {
        let num = i + 1;
        println!("\n****Running Point {}/{}****", num, geometries.len());

        // Run state A
        match qm.run_calculation(Path::new(&format!("running_dir/{}_A.gjf", num))) {
            Ok(_) => {
                match qm.read_output(
                    Path::new(&format!("running_dir/{}_A.log", num)),
                    config.state1,
                ) {
                    Ok(state_a) => {
                        energies_a.push(state_a.energy);
                        println!("✓ State A completed: E = {:.8} hartree", state_a.energy);
                    }
                    Err(e) => {
                        println!("✗ Failed to read state A output: {}", e);
                        energies_a.push(f64::NAN);
                    }
                }
            }
            Err(e) => {
                println!("✗ Failed to run state A: {}", e);
                energies_a.push(f64::NAN);
                continue;
            }
        }

        // Run state B
        match qm.run_calculation(Path::new(&format!("running_dir/{}_B.gjf", num))) {
            Ok(_) => {
                match qm.read_output(
                    Path::new(&format!("running_dir/{}_B.log", num)),
                    config.state2,
                ) {
                    Ok(state_b) => {
                        energies_b.push(state_b.energy);
                        println!("✓ State B completed: E = {:.8} hartree", state_b.energy);
                    }
                    Err(e) => {
                        println!("✗ Failed to read state B output: {}", e);
                        energies_b.push(f64::NAN);
                    }
                }
            }
            Err(e) => {
                println!("✗ Failed to run state B: {}", e);
                energies_b.push(f64::NAN);
                continue;
            }
        }

        if !energies_a.last().unwrap().is_nan() && !energies_b.last().unwrap().is_nan() {
            successful_points += 1;
        }
    }

    // Print energy profile
    println!("\n****Energy Profile****");
    println!(
        "Successful points: {}/{}",
        successful_points,
        geometries.len()
    );

    let mut min_de = f64::INFINITY;
    let mut min_de_idx = 0;

    for (i, (ea, eb)) in energies_a.iter().zip(energies_b.iter()).enumerate() {
        if ea.is_finite() && eb.is_finite() {
            let de = ea - eb;
            println!(
                "Point {:2}: EA = {:.8}, EB = {:.8}, ΔE = {:.8}",
                i + 1,
                ea,
                eb,
                de
            );

            if de.abs() < min_de.abs() {
                min_de = de;
                min_de_idx = i;
            }
        } else {
            println!("Point {:2}: Calculation failed", i + 1);
        }
    }

    if successful_points > 0 {
        println!("\n****Summary****");
        println!(
            "Minimum |ΔE| found at point {}: {:.8} hartree ({:.3} eV)",
            min_de_idx + 1,
            min_de,
            min_de * 27.211386
        );
        println!(
            "Suggested MECP starting geometry: running_dir/{}_A.gjf",
            min_de_idx + 1
        );
    }

    Ok(())
}

fn run_pre_point(
    geometry: &geometry::Geometry,
    header_a: &str,
    header_b: &str,
    input_data: &parser::InputData,
    qm: &dyn qm_interface::QMInterface,
    run_mode: config::RunMode,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running pre-point calculations****");

    // Write and run state B first
    qm.write_input(
        geometry,
        header_b,
        &input_data.tail2,
        Path::new("running_dir/pre_B.gjf"),
    )?;
    qm.run_calculation(Path::new("running_dir/pre_B.gjf"))?;

    // For inter_read mode, copy B checkpoint to A
    if run_mode == config::RunMode::InterRead {
        println!("Inter-read mode: copying state B wavefunction to state A");
        std::fs::copy("b.chk", "a.chk")?;

        // For Gaussian, add guess=mix to state A
        let mut header_a_modified = header_a.to_string();
        if header_a_modified.contains("guess=read") {
            header_a_modified = header_a_modified.replace("guess=read", "guess=(read,mix)");
        }
        qm.write_input(
            geometry,
            &header_a_modified,
            &input_data.tail1,
            Path::new("running_dir/pre_A.gjf"),
        )?;
    } else {
        qm.write_input(
            geometry,
            header_a,
            &input_data.tail1,
            Path::new("running_dir/pre_A.gjf"),
        )?;
    }

    // Run state A
    qm.run_calculation(Path::new("running_dir/pre_A.gjf"))?;

    println!("Pre-point calculations complete");
    Ok(())
}

fn run_restart(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Restarting from checkpoint****");

    // Load checkpoint
    let checkpoint_path = Path::new(&input_data.config.checkpoint_file);
    let (step, mut geometry, x_old, hessian, mut opt_state, config) =
        checkpoint::Checkpoint::load(checkpoint_path)?;

    println!("Loaded checkpoint from step {}", step);

    let constraints = &input_data.constraints;
    let fixed_atoms = &input_data.fixed_atoms;

    // Build headers
    let header_a = io::build_gaussian_header(&config, config.charge1, config.mult1, &config.td1);
    let header_b = io::build_gaussian_header(&config, config.charge2, config.mult2, &config.td2);

    // Continue optimization from the next step
    let start_step = step + 1;
    for step in start_step..config.max_steps {
        println!("\n****Step {}****", step + 1);

        // Run calculations with current geometry
        let step_name = format!("running_dir/{}_A.gjf", step);
        qm.write_input(
            &geometry,
            &header_a,
            &input_data.tail1,
            Path::new(&step_name),
        )?;
        qm.run_calculation(Path::new(&step_name))?;

        let step_name = format!("running_dir/{}_B.gjf", step);
        qm.write_input(
            &geometry,
            &header_b,
            &input_data.tail2,
            Path::new(&step_name),
        )?;
        qm.run_calculation(Path::new(&step_name))?;

        let state1 = qm.read_output(
            Path::new(&format!("running_dir/{}_A.log", step)),
            config.state1,
        )?;
        let state2 = qm.read_output(
            Path::new(&format!("running_dir/{}_B.log", step)),
            config.state2,
        )?;

        // Compute MECP gradient
        let grad = optimizer::compute_mecp_gradient(
            &state1,
            &state2,
            constraints,
            &mut opt_state,
            fixed_atoms,
        );

        // Choose optimizer
        let x_new = if step < 3 || !opt_state.has_enough_history() {
            println!("Using BFGS optimizer");
            optimizer::bfgs_step(&x_old, &grad, &hessian, &config)
        } else {
            println!("Using GDIIS optimizer");
            optimizer::gdiis_step(&opt_state, &config)
        };

        // Update geometry
        geometry.coords = x_new.clone();

        // Check convergence
        let conv = optimizer::check_convergence(
            state1.energy,
            state2.energy,
            &x_old,
            &x_new,
            &grad,
            &config,
        );

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state1.energy,
            state2.energy,
            (state1.energy - state2.energy).abs()
        );

        if conv.is_converged() {
            println!("\n****Congrats! MECP has converged****");
            println!("Final geometry saved to final.xyz");
            io::write_xyz(&geometry, Path::new("final.xyz"))?;
            return Ok(());
        }

        // Check convergence
        let conv = optimizer::check_convergence(
            state1.energy,
            state2.energy,
            &x_old,
            &x_new,
            &grad,
            &config,
        );

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state1.energy,
            state2.energy,
            (state1.energy - state2.energy).abs()
        );

        if conv.is_converged() {
            println!("\n****Congrats! MECP has converged****");
            println!("Final geometry saved to final.xyz");
            io::write_xyz(&geometry, Path::new("final.xyz"))?;
            return Ok(());
        }

        if step >= config.max_steps {
            break;
        }
    }

    Err("Maximum steps exceeded".into())
}

fn run_coordinate_driving(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Coordinate Driving****");

    let config = &input_data.config;
    let geometry = &input_data.geometry;

    // Parse drive coordinate
    if config.drive_atoms.is_empty() || config.drive_type.is_empty() {
        return Err("Coordinate driving requires drive_type and drive_atoms parameters".into());
    }

    let coord_type = match config.drive_type.as_str() {
        "bond" => reaction_path::CoordinateType::Bond,
        "angle" => reaction_path::CoordinateType::Angle,
        "dihedral" => reaction_path::CoordinateType::Dihedral,
        _ => return Err(format!("Unknown drive_type: {}", config.drive_type).into()),
    };

    println!("Driving coordinate: {:?}", coord_type);

    let drive_coord = reaction_path::DriveCoordinate::new(
        coord_type,
        config.drive_atoms.clone(),
        config.drive_start,
    );
    println!(
        "Atoms: {:?}",
        config
            .drive_atoms
            .iter()
            .map(|&x| x + 1)
            .collect::<Vec<_>>()
    );
    println!(
        "From {:.3} to {:.3} in {} steps",
        config.drive_start, config.drive_end, config.drive_steps
    );

    // Generate reaction path
    let path = reaction_path::drive_coordinate(
        geometry,
        &drive_coord,
        config.drive_start,
        config.drive_end,
        config.drive_steps,
    );

    println!(
        "Generated {} geometries along reaction coordinate",
        path.len()
    );

    // Build headers
    let header_a = io::build_gaussian_header(config, config.charge1, config.mult1, &config.td1);
    let header_b = io::build_gaussian_header(config, config.charge2, config.mult2, &config.td2);

    // Run calculations along the path
    let mut energies_a = Vec::new();
    let mut energies_b = Vec::new();

    for (i, geom) in path.iter().enumerate() {
        let step = i + 1;
        println!("\n****Step {}/{}****", step, path.len());

        // Current coordinate value
        let current_value = drive_coord.current_value(geom);
        println!("Coordinate value: {:.3}", current_value);

        // Write input files
        qm.write_input(
            geom,
            &header_a,
            &input_data.tail1,
            Path::new(&format!("running_dir/drive_{}_A.gjf", step)),
        )?;
        qm.write_input(
            geom,
            &header_b,
            &input_data.tail2,
            Path::new(&format!("running_dir/drive_{}_B.gjf", step)),
        )?;

        // Run calculations
        qm.run_calculation(Path::new(&format!("running_dir/drive_{}_A.gjf", step)))?;
        qm.run_calculation(Path::new(&format!("running_dir/drive_{}_B.gjf", step)))?;

        // Read results
        let state_a = qm.read_output(
            Path::new(&format!("running_dir/drive_{}_A.log", step)),
            config.state1,
        )?;
        let state_b = qm.read_output(
            Path::new(&format!("running_dir/drive_{}_B.log", step)),
            config.state2,
        )?;

        energies_a.push(state_a.energy);
        energies_b.push(state_b.energy);

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state_a.energy,
            state_b.energy,
            state_a.energy - state_b.energy
        );
    }

    // Print energy profile
    println!("\n****Energy Profile Along Reaction Coordinate****");
    println!("Step | Coordinate |    E1    |    E2    |   ΔE   ");
    println!("-----|------------|----------|----------|--------");

    for (i, (&ea, &eb)) in energies_a.iter().zip(energies_b.iter()).enumerate() {
        let coord_value = config.drive_start
            + (config.drive_end - config.drive_start) * (i as f64)
                / ((config.drive_steps - 1) as f64);
        println!(
            "{:4} | {:>10.3} | {:>8.3} | {:>8.3} | {:>7.3}",
            i + 1,
            coord_value,
            ea,
            eb,
            ea - eb
        );
    }

    // Analyze path
    let stats = reaction_path::analyze_reaction_path(&path);
    println!("\n****Path Statistics****");
    println!("Total path length: {:.3} Å", stats.path_length);
    println!("Number of points: {}", stats.num_points);

    Ok(())
}

fn run_path_optimization(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Path Optimization****");

    let config = &input_data.config;
    let geometry = &input_data.geometry;

    // For path optimization, we need an initial path
    // This could come from coordinate driving or be provided as multiple geometries
    // For now, we'll create a simple initial path using coordinate driving

    if config.drive_atoms.is_empty() || config.drive_type.is_empty() {
        return Err("Path optimization requires drive_type and drive_atoms parameters to create initial path".into());
    }

    let coord_type = match config.drive_type.as_str() {
        "bond" => reaction_path::CoordinateType::Bond,
        "angle" => reaction_path::CoordinateType::Angle,
        "dihedral" => reaction_path::CoordinateType::Dihedral,
        _ => return Err(format!("Unknown drive_type: {}", config.drive_type).into()),
    };

    let drive_coord = reaction_path::DriveCoordinate::new(
        coord_type,
        config.drive_atoms.clone(),
        config.drive_start,
    );

    println!("Creating initial path via coordinate driving...");
    let initial_path = reaction_path::drive_coordinate(
        geometry,
        &drive_coord,
        config.drive_start,
        config.drive_end,
        config.drive_steps,
    );

    println!("Initial path created with {} points", initial_path.len());

    // Optimize the path using NEB
    println!("Optimizing path using Nudged Elastic Band (NEB) method...");
    let optimized_path =
        reaction_path::optimize_reaction_path(&initial_path, &input_data.constraints);

    println!(
        "Path optimization completed. Optimized path has {} points",
        optimized_path.len()
    );

    // Build headers for QM calculations
    let header_a = io::build_gaussian_header(config, config.charge1, config.mult1, &config.td1);
    let header_b = io::build_gaussian_header(config, config.charge2, config.mult2, &config.td2);

    // Run calculations along the optimized path
    let mut energies_a = Vec::new();
    let mut energies_b = Vec::new();

    for (i, geom) in optimized_path.iter().enumerate() {
        let step = i + 1;
        println!(
            "\n****Optimized Path Step {}/{}****",
            step,
            optimized_path.len()
        );

        // Write input files
        qm.write_input(
            geom,
            &header_a,
            &input_data.tail1,
            Path::new(&format!("running_dir/neb_{}_A.gjf", step)),
        )?;
        qm.write_input(
            geom,
            &header_b,
            &input_data.tail2,
            Path::new(&format!("running_dir/neb_{}_B.gjf", step)),
        )?;

        // Run calculations
        qm.run_calculation(Path::new(&format!("running_dir/neb_{}_A.gjf", step)))?;
        qm.run_calculation(Path::new(&format!("running_dir/neb_{}_B.gjf", step)))?;

        // Read results
        let state_a = qm.read_output(
            Path::new(&format!("running_dir/neb_{}_A.log", step)),
            config.state1,
        )?;
        let state_b = qm.read_output(
            Path::new(&format!("running_dir/neb_{}_B.log", step)),
            config.state2,
        )?;

        energies_a.push(state_a.energy);
        energies_b.push(state_b.energy);

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}",
            state_a.energy,
            state_b.energy,
            state_a.energy - state_b.energy
        );
    }

    // Print optimized energy profile
    println!("\n****Optimized Path Energy Profile****");
    println!("Step |    E1    |    E2    |   ΔE   ");
    println!("-----|----------|----------|--------");

    for (i, (&ea, &eb)) in energies_a.iter().zip(energies_b.iter()).enumerate() {
        println!("{:4} | {:>8.3} | {:>8.3} | {:>7.3}", i + 1, ea, eb, ea - eb);
    }

    // Analyze optimized path
    let stats = reaction_path::analyze_reaction_path(&optimized_path);
    println!("\n****Optimized Path Statistics****");
    println!("Total path length: {:.3} Å", stats.path_length);
    println!("Number of points: {}", stats.num_points);

    Ok(())
}

fn run_fix_de(
    input_data: parser::InputData,
    qm: &dyn qm_interface::QMInterface,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n****Running Fix-dE Optimization****");

    let config = &input_data.config;
    let mut geometry = input_data.geometry;
    let constraints = &input_data.constraints;
    let fixed_atoms = &input_data.fixed_atoms;

    if config.fix_de == 0.0 {
        return Err("Fix-dE optimization requires fix_de parameter to be set".into());
    }

    // Build headers
    let header_a = io::build_gaussian_header(config, config.charge1, config.mult1, &config.td1);
    let header_b = io::build_gaussian_header(config, config.charge2, config.mult2, &config.td2);

    // Initialize optimization
    let mut opt_state = optimizer::OptimizationState::new();
    let mut x_old = geometry.coords.clone();
    let mut hessian = DMatrix::identity(geometry.coords.len(), geometry.coords.len());

    // Main optimization loop
    for step in 0..config.max_steps {
        println!("\n****Fix-dE Step {}****", step + 1);

        // Run calculations
        let step_name_a = format!("running_dir/{}_A.gjf", step + 1);
        let step_name_b = format!("running_dir/{}_B.gjf", step + 1);

        qm.write_input(
            &geometry,
            &header_a,
            &input_data.tail1,
            Path::new(&step_name_a),
        )?;
        qm.write_input(
            &geometry,
            &header_b,
            &input_data.tail2,
            Path::new(&step_name_b),
        )?;

        qm.run_calculation(Path::new(&step_name_a))?;
        qm.run_calculation(Path::new(&step_name_b))?;

        let state1 = qm.read_output(
            Path::new(&format!("running_dir/{}_A.log", step + 1)),
            config.state1,
        )?;
        let state2 = qm.read_output(
            Path::new(&format!("running_dir/{}_B.log", step + 1)),
            config.state2,
        )?;

        // Compute MECP gradient with energy difference constraint
        let grad = optimizer::compute_mecp_gradient_with_de_constraint(
            &state1,
            &state2,
            constraints,
            &mut opt_state,
            fixed_atoms,
            config.fix_de,
        );

        // Choose optimizer
        let x_new = if step < 3 || !opt_state.has_enough_history() {
            println!("Using BFGS optimizer");
            optimizer::bfgs_step(&x_old, &grad, &hessian, config)
        } else if config.use_gediis {
            println!("Using GEDIIS optimizer");
            optimizer::gediis_step(&opt_state, config)
        } else {
            println!("Using GDIIS optimizer");
            optimizer::gdiis_step(&opt_state, config)
        };

        // Update geometry
        geometry.coords = x_new.clone();

        // Check convergence
        let de = (state1.energy - state2.energy).abs();
        let target_de = config.fix_de.abs();
        let de_error = (de - target_de).abs();

        println!(
            "E1 = {:.8}, E2 = {:.8}, ΔE = {:.8}, Target ΔE = {:.8}, ΔE Error = {:.8}",
            state1.energy, state2.energy, de, target_de, de_error
        );

        // Check if energy difference is close to target
        if de_error < config.thresholds.de {
            println!("\nFix-dE optimization converged at step {}", step + 1);
            io::write_xyz(&geometry, Path::new("final.xyz"))?;
            return Ok(());
        }

        // Update Hessian
        let sk = &x_new - &x_old;
        let yk = &grad
            - &optimizer::compute_mecp_gradient_with_de_constraint(
                &qm.read_output(
                    Path::new(&format!("running_dir/{}_A.log", step)),
                    config.state1,
                )?,
                &qm.read_output(
                    Path::new(&format!("running_dir/{}_B.log", step)),
                    config.state2,
                )?,
                constraints,
                &mut opt_state,
                fixed_atoms,
                config.fix_de,
            );
        hessian = optimizer::update_hessian_psb(&hessian, &sk, &yk);

        // Add to history
        let energy_diff = state1.energy - state2.energy;
        opt_state.add_to_history(x_new.clone(), grad.clone(), hessian.clone(), energy_diff);

        x_old = x_new;
    }

    Err("Maximum steps exceeded in fix-dE optimization".into())
}
