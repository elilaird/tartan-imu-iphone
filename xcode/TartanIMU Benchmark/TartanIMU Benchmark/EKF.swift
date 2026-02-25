import Foundation
import Accelerate

/// Extended Kalman Filter for fusing TartanIMU velocity estimates.
///
/// State: [px, py, pz, vx, vy, vz] (6D)
/// Measurement: [vx, vy, vz] from TartanIMU + [log_σx, log_σy, log_σz] covariance
///
/// Prediction step: constant-velocity model (position += velocity * dt)
/// Update step: velocity measurement from neural network with learned covariance
class EKF {

    /// State dimension
    static let n = 6
    /// Measurement dimension
    static let m = 3

    /// State vector [px, py, pz, vx, vy, vz]
    var x: [Float]

    /// Covariance matrix (n x n, row-major)
    var P: [Float]

    /// Process noise covariance (n x n, row-major)
    let Q: [Float]

    /// Time step (seconds)
    let dt: Float

    init(dt: Float = 1.0, processNoisePosStd: Float = 0.1, processNoiseVelStd: Float = 0.5) {
        self.dt = dt

        // Initial state: zeros
        x = [Float](repeating: 0, count: EKF.n)

        // Initial covariance: large uncertainty
        P = EKF.diag(EKF.n, values: [10, 10, 10, 1, 1, 1])

        // Process noise: position grows with velocity uncertainty, velocity has its own noise
        var q = [Float](repeating: 0, count: EKF.n * EKF.n)
        let qp = processNoisePosStd * processNoisePosStd
        let qv = processNoiseVelStd * processNoiseVelStd
        for i in 0..<3 { q[i * EKF.n + i] = qp }         // position
        for i in 3..<6 { q[i * EKF.n + i] = qv }         // velocity
        Q = q
    }

    /// Predict step: propagate state with constant-velocity model.
    func predict() {
        // x_new = F * x  where F is identity + dt in upper-right 3x3 block
        // pos += vel * dt
        x[0] += x[3] * dt
        x[1] += x[4] * dt
        x[2] += x[5] * dt
        // velocity unchanged in prediction

        // P_new = F * P * F^T + Q
        let F = stateTransitionMatrix()
        let FP = EKF.matMul(F, P, EKF.n, EKF.n, EKF.n)
        let Ft = EKF.transpose(F, EKF.n, EKF.n)
        let FPFt = EKF.matMul(FP, Ft, EKF.n, EKF.n, EKF.n)
        P = EKF.matAdd(FPFt, Q)
    }

    /// Update step: incorporate TartanIMU velocity measurement.
    ///
    /// - Parameters:
    ///   - velocityMeasurement: [vx, vy, vz] from model
    ///   - logCovariance: [log_σx, log_σy, log_σz] from model (diagonal measurement noise)
    func update(velocityMeasurement z: [Float], logCovariance: [Float]) {
        // Measurement model: H * x = [vx, vy, vz]
        // H is [0 0 0 1 0 0; 0 0 0 0 1 0; 0 0 0 0 0 1]
        let H = measurementMatrix()

        // Measurement noise R from learned log-covariance
        // σ² = exp(log_σ), so R_ii = exp(log_σ_i)
        var R = [Float](repeating: 0, count: EKF.m * EKF.m)
        for i in 0..<3 {
            R[i * EKF.m + i] = exp(logCovariance[i])
        }

        // Innovation: y = z - H * x
        let Hx = EKF.matVecMul(H, x, EKF.m, EKF.n)
        var y = [Float](repeating: 0, count: EKF.m)
        for i in 0..<EKF.m { y[i] = z[i] - Hx[i] }

        // Innovation covariance: S = H * P * H^T + R
        let HP = EKF.matMul(H, P, EKF.m, EKF.n, EKF.n)
        let Ht = EKF.transpose(H, EKF.m, EKF.n)
        let HPHt = EKF.matMul(HP, Ht, EKF.m, EKF.n, EKF.m)
        let S = EKF.matAdd(HPHt, R)

        // Kalman gain: K = P * H^T * S^-1
        let PHt = EKF.matMul(P, Ht, EKF.n, EKF.n, EKF.m)
        let Sinv = EKF.invert3x3(S)
        let K = EKF.matMul(PHt, Sinv, EKF.n, EKF.m, EKF.m)

        // State update: x = x + K * y
        let Ky = EKF.matVecMul(K, y, EKF.n, EKF.m)
        for i in 0..<EKF.n { x[i] += Ky[i] }

        // Covariance update: P = (I - K * H) * P
        let KH = EKF.matMul(K, H, EKF.n, EKF.m, EKF.n)
        var IKH = EKF.identity(EKF.n)
        for i in 0..<EKF.n * EKF.n { IKH[i] -= KH[i] }
        P = EKF.matMul(IKH, P, EKF.n, EKF.n, EKF.n)
    }

    /// Current position estimate.
    var position: (x: Float, y: Float, z: Float) {
        (x[0], x[1], x[2])
    }

    /// Current velocity estimate.
    var velocity: (x: Float, y: Float, z: Float) {
        (x[3], x[4], x[5])
    }

    /// Reset filter to initial state.
    func reset() {
        x = [Float](repeating: 0, count: EKF.n)
        P = EKF.diag(EKF.n, values: [10, 10, 10, 1, 1, 1])
    }

    // MARK: - Matrices

    private func stateTransitionMatrix() -> [Float] {
        var F = EKF.identity(EKF.n)
        // pos += vel * dt
        F[0 * EKF.n + 3] = dt  // px += vx * dt
        F[1 * EKF.n + 4] = dt  // py += vy * dt
        F[2 * EKF.n + 5] = dt  // pz += vz * dt
        return F
    }

    private func measurementMatrix() -> [Float] {
        // H: 3x6 — observes velocity components
        var H = [Float](repeating: 0, count: EKF.m * EKF.n)
        H[0 * EKF.n + 3] = 1  // observe vx
        H[1 * EKF.n + 4] = 1  // observe vy
        H[2 * EKF.n + 5] = 1  // observe vz
        return H
    }

    // MARK: - Linear Algebra Helpers

    static func identity(_ n: Int) -> [Float] {
        var I = [Float](repeating: 0, count: n * n)
        for i in 0..<n { I[i * n + i] = 1 }
        return I
    }

    static func diag(_ n: Int, values: [Float]) -> [Float] {
        var D = [Float](repeating: 0, count: n * n)
        for i in 0..<min(n, values.count) { D[i * n + i] = values[i] }
        return D
    }

    static func matMul(_ A: [Float], _ B: [Float], _ m: Int, _ k: Int, _ n: Int) -> [Float] {
        var C = [Float](repeating: 0, count: m * n)
        vDSP_mmul(A, 1, B, 1, &C, 1, vDSP_Length(m), vDSP_Length(n), vDSP_Length(k))
        return C
    }

    static func matVecMul(_ A: [Float], _ x: [Float], _ m: Int, _ n: Int) -> [Float] {
        var y = [Float](repeating: 0, count: m)
        vDSP_mmul(A, 1, x, 1, &y, 1, vDSP_Length(m), vDSP_Length(1), vDSP_Length(n))
        return y
    }

    static func transpose(_ A: [Float], _ m: Int, _ n: Int) -> [Float] {
        var At = [Float](repeating: 0, count: n * m)
        vDSP_mtrans(A, 1, &At, 1, vDSP_Length(n), vDSP_Length(m))
        return At
    }

    static func matAdd(_ A: [Float], _ B: [Float]) -> [Float] {
        var C = [Float](repeating: 0, count: A.count)
        vDSP_vadd(A, 1, B, 1, &C, 1, vDSP_Length(A.count))
        return C
    }

    /// Invert a 3x3 matrix using Cramer's rule.
    static func invert3x3(_ M: [Float]) -> [Float] {
        let a = M[0], b = M[1], c = M[2]
        let d = M[3], e = M[4], f = M[5]
        let g = M[6], h = M[7], k = M[8]

        let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g)
        guard abs(det) > 1e-10 else { return identity(3) }

        let invDet = 1.0 / det
        return [
            (e * k - f * h) * invDet, (c * h - b * k) * invDet, (b * f - c * e) * invDet,
            (f * g - d * k) * invDet, (a * k - c * g) * invDet, (c * d - a * f) * invDet,
            (d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet,
        ]
    }
}
