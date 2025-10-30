use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub enum Constraint {
    Bond { atoms: (usize, usize), target: f64 },
    Angle { atoms: (usize, usize, usize), target: f64 },
    Dihedral { atoms: (usize, usize, usize, usize), target: f64 },
}

pub fn evaluate_bond(coords: &DVector<f64>, a: usize, b: usize) -> f64 {
    let i = a * 3;
    let j = b * 3;
    let dx = coords[j] - coords[i];
    let dy = coords[j + 1] - coords[i + 1];
    let dz = coords[j + 2] - coords[i + 2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub fn evaluate_angle(coords: &DVector<f64>, a: usize, b: usize, c: usize) -> f64 {
    let ia = a * 3;
    let ib = b * 3;
    let ic = c * 3;

    let bax = coords[ia] - coords[ib];
    let bay = coords[ia + 1] - coords[ib + 1];
    let bazz = coords[ia + 2] - coords[ib + 2];

    let bcx = coords[ic] - coords[ib];
    let bcy = coords[ic + 1] - coords[ib + 1];
    let bcz = coords[ic + 2] - coords[ib + 2];

    let rba = (bax * bax + bay * bay + bazz * bazz).sqrt();
    let rbc = (bcx * bcx + bcy * bcy + bcz * bcz).sqrt();
    let dot = bax * bcx + bay * bcy + bazz * bcz;

    (dot / (rba * rbc)).acos()
}

pub fn evaluate_dihedral(_coords: &DVector<f64>, _a: usize, _b: usize, _c: usize, _d: usize) -> f64 {
    // Calculate dihedral angle between planes ABC and BCD
    // This is a simplified implementation - proper dihedral calculation is more complex
    // For now, return 0.0 as placeholder
    0.0
}

/// Compute gradient of bond constraint: dR/dx
pub fn bond_gradient(coords: &DVector<f64>, a: usize, b: usize) -> DVector<f64> {
    let mut grad = DVector::zeros(coords.len());
    
    let i = a * 3;
    let j = b * 3;
    
    let dx = coords[j] - coords[i];
    let dy = coords[j + 1] - coords[i + 1];
    let dz = coords[j + 2] - coords[i + 2];
    let r = (dx * dx + dy * dy + dz * dz).sqrt();
    
    if r > 1e-10 {
        grad[i] = -dx / r;
        grad[i + 1] = -dy / r;
        grad[i + 2] = -dz / r;
        
        grad[j] = dx / r;
        grad[j + 1] = dy / r;
        grad[j + 2] = dz / r;
    }
    
    grad
}

/// Compute gradient of angle constraint: dA/dx
pub fn angle_gradient(coords: &DVector<f64>, a: usize, b: usize, c: usize) -> DVector<f64> {
    let mut grad = DVector::zeros(coords.len());

    let ia = a * 3;
    let ib = b * 3;
    let ic = c * 3;

    let bax = coords[ia] - coords[ib];
    let bay = coords[ia + 1] - coords[ib + 1];
    let bazz = coords[ia + 2] - coords[ib + 2];

    let bcx = coords[ic] - coords[ib];
    let bcy = coords[ic + 1] - coords[ib + 1];
    let bcz = coords[ic + 2] - coords[ib + 2];

    let rba = (bax * bax + bay * bay + bazz * bazz).sqrt();
    let rbc = (bcx * bcx + bcy * bcy + bcz * bcz).sqrt();
    let dot = bax * bcx + bay * bcy + bazz * bcz;

    if rba > 1e-10 && rbc > 1e-10 {
        grad[ia] = (rba * bcx - dot * bax / rba) / (rbc * rba * rba);
        grad[ia + 1] = (rba * bcy - dot * bay / rba) / (rbc * rba * rba);
        grad[ia + 2] = (rba * bcz - dot * bazz / rba) / (rbc * rba * rba);

        grad[ic] = (rbc * bax - dot * bcx / rbc) / (rba * rbc * rbc);
        grad[ic + 1] = (rbc * bay - dot * bcy / rbc) / (rba * rbc * rbc);
        grad[ic + 2] = (rbc * bazz - dot * bcz / rbc) / (rba * rbc * rbc);

        grad[ib] = -(grad[ia] + grad[ic]);
        grad[ib + 1] = -(grad[ia + 1] + grad[ic + 1]);
        grad[ib + 2] = -(grad[ia + 2] + grad[ic + 2]);
    }

    grad
}

pub fn dihedral_gradient(coords: &DVector<f64>, _a: usize, _b: usize, _c: usize, _d: usize) -> DVector<f64> {
    // Placeholder implementation - proper dihedral gradient is complex
    DVector::zeros(coords.len())
}

/// Build Jacobian matrix: C[i,j] = dConstraint[i]/dx[j]
pub fn build_jacobian(coords: &DVector<f64>, constraints: &[Constraint]) -> DMatrix<f64> {
    let n_coords = coords.len();
    let n_constraints = constraints.len();
    
    let mut jacobian = DMatrix::zeros(n_constraints, n_coords);
    
    for (i, constraint) in constraints.iter().enumerate() {
        let grad = match constraint {
            Constraint::Bond { atoms: (a, b), .. } => bond_gradient(coords, *a, *b),
            Constraint::Angle { atoms: (a, b, c), .. } => angle_gradient(coords, *a, *b, *c),
            Constraint::Dihedral { atoms: (a, b, c, d), .. } => dihedral_gradient(coords, *a, *b, *c, *d),
        };
        
        jacobian.set_row(i, &grad.transpose());
    }
    
    jacobian
}

/// Evaluate constraint violations
pub fn evaluate_constraints(coords: &DVector<f64>, constraints: &[Constraint]) -> DVector<f64> {
    let mut violations = DVector::zeros(constraints.len());
    
    for (i, constraint) in constraints.iter().enumerate() {
        let current = match constraint {
            Constraint::Bond { atoms: (a, b), target } => evaluate_bond(coords, *a, *b) - target,
            Constraint::Angle { atoms: (a, b, c), target } => evaluate_angle(coords, *a, *b, *c) - target,
            Constraint::Dihedral { atoms: (a, b, c, d), target } => evaluate_dihedral(coords, *a, *b, *c, *d) - target,
        };
        violations[i] = current;
    }
    
    violations
}
