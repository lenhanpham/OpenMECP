use omecp::gdiis::{CosineCheckMode, GdiisOptimizer, TriangularMatrix, variable_cos_limit};

#[test]
fn test_triangular_matrix() {
    let mut m = TriangularMatrix::new(3);
    m.set(0, 0, 1.0);
    m.set(1, 0, 2.0);
    m.set(1, 1, 3.0);
    m.set(2, 0, 4.0);
    m.set(2, 1, 5.0);
    m.set(2, 2, 6.0);

    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(1, 0), 2.0);
    assert_eq!(m.get(0, 1), 2.0); // Symmetric access
    assert_eq!(m.get(2, 2), 6.0);
}

#[test]
fn test_triangular_matrix_multiply() {
    let mut m = TriangularMatrix::new(2);
    m.set(0, 0, 1.0);
    m.set(1, 0, 0.0);
    m.set(1, 1, 1.0);

    let x = vec![2.0, 3.0];
    let y = m.multiply_vector(&x);

    assert_eq!(y[0], 2.0);
    assert_eq!(y[1], 3.0);
}

#[test]
fn test_variable_cos_limit() {
    assert_eq!(variable_cos_limit(2), 0.97);
    assert_eq!(variable_cos_limit(4), 0.71);
    assert_eq!(variable_cos_limit(10), 0.0);
}

#[test]
fn test_gdiis_optimizer_creation() {
    let opt = GdiisOptimizer::new(5);
    assert_eq!(opt.max_vectors, 5);
    assert_eq!(opt.cosine_check, CosineCheckMode::Standard);
}
