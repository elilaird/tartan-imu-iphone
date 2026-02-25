import CoreML
import Foundation
import QuartzCore

/// Runs TartanIMU CoreML inference on live IMU data and tracks performance metrics.
class TartanIMURunner: ObservableObject {

    private var model: MLModel?
    private let imuCapture: IMUCapture

    // LSTM state — persisted across calls
    private var hiddenState: MLMultiArray
    private var cellState: MLMultiArray

    // EKF
    private let ekf = EKF(dt: 1.0)

    // --- Performance metrics ---

    /// Rolling window of per-call latencies (model only)
    @Published var latencyHistory: [Double] = []
    /// Rolling window of per-call latencies (model + EKF)
    @Published var ekfLatencyHistory: [Double] = []
    /// Rolling window of IMU sample rate measurements
    @Published var sampleRateHistory: [Double] = []

    /// Current values
    @Published var inferenceTimeMs: Double = 0
    @Published var ekfInferenceTimeMs: Double = 0
    @Published var throughputFPS: Double = 0
    @Published var sampleRate: Double = 0

    /// Aggregate stats
    @Published var minLatencyMs: Double = .infinity
    @Published var maxLatencyMs: Double = 0
    @Published var meanLatencyMs: Double = 0
    @Published var p99LatencyMs: Double = 0
    @Published var ekfOverheadMs: Double = 0
    @Published var totalInferences: Int = 0
    @Published var droppedWindows: Int = 0

    @Published var isRunning = false

    private let historySize = 120
    private var allLatencies: [Double] = []
    private var inferenceInProgress = false

    init() {
        hiddenState = try! MLMultiArray(shape: [2, 1, 512], dataType: .float32)
        cellState = try! MLMultiArray(shape: [2, 1, 512], dataType: .float32)
        for i in 0..<hiddenState.count { hiddenState[i] = 0 }
        for i in 0..<cellState.count { cellState[i] = 0 }

        imuCapture = IMUCapture(windowSize: 200, sampleRate: 100.0)

        loadModel()

        imuCapture.onWindowReady = { [weak self] window in
            self?.runInference(window: window)
        }
    }

    private func loadModel() {
        let config = MLModelConfiguration()
        config.computeUnits = .all

        guard let modelURL = Bundle.main.url(forResource: "tartanimu_car", withExtension: "mlmodelc") ??
                             Bundle.main.url(forResource: "tartanimu_car", withExtension: "mlpackage") else {
            print("Model not found in bundle. Add tartanimu_car.mlpackage to the Xcode project.")
            return
        }

        do {
            model = try MLModel(contentsOf: modelURL, configuration: config)
            print("Model loaded successfully")
        } catch {
            print("Failed to load model: \(error)")
        }
    }

    func start() {
        reset()
        imuCapture.start()
        DispatchQueue.main.async { self.isRunning = true }
    }

    func stop() {
        imuCapture.stop()
        DispatchQueue.main.async { self.isRunning = false }
    }

    func reset() {
        for i in 0..<hiddenState.count { hiddenState[i] = 0 }
        for i in 0..<cellState.count { cellState[i] = 0 }
        ekf.reset()
        allLatencies = []
        inferenceInProgress = false
        DispatchQueue.main.async {
            self.latencyHistory = []
            self.ekfLatencyHistory = []
            self.sampleRateHistory = []
            self.inferenceTimeMs = 0
            self.ekfInferenceTimeMs = 0
            self.throughputFPS = 0
            self.sampleRate = 0
            self.minLatencyMs = .infinity
            self.maxLatencyMs = 0
            self.meanLatencyMs = 0
            self.p99LatencyMs = 0
            self.ekfOverheadMs = 0
            self.totalInferences = 0
            self.droppedWindows = 0
        }
    }

    private func runInference(window: [[Float]]) {
        guard let model else { return }

        // Track if we're falling behind (previous inference still running)
        if inferenceInProgress {
            DispatchQueue.main.async { self.droppedWindows += 1 }
            return
        }
        inferenceInProgress = true

        // Pack buffer into MLMultiArray [1, 6, 200]
        let imuInput = try! MLMultiArray(shape: [1, 6, 200], dataType: .float32)
        for t in 0..<200 {
            let sample = window[t]
            for ch in 0..<6 {
                imuInput[[0, ch, t] as [NSNumber]] = NSNumber(value: sample[ch])
            }
        }

        let inputFeatures = try! MLDictionaryFeatureProvider(dictionary: [
            "imu_window": MLFeatureValue(multiArray: imuInput),
            "hidden": MLFeatureValue(multiArray: hiddenState),
            "cell": MLFeatureValue(multiArray: cellState),
        ])

        // Time model inference only
        let t0 = CACurrentMediaTime()
        guard let output = try? model.prediction(from: inputFeatures) else {
            print("Inference failed")
            inferenceInProgress = false
            return
        }
        let modelMs = (CACurrentMediaTime() - t0) * 1000

        // Update LSTM state
        hiddenState = output.featureValue(for: "hidden_out")!.multiArrayValue!
        cellState = output.featureValue(for: "cell_out")!.multiArrayValue!

        // Time model + EKF
        let t1 = CACurrentMediaTime()
        let vel = output.featureValue(for: "velocity")!.multiArrayValue!
        let lcov = output.featureValue(for: "log_covariance")!.multiArrayValue!
        let z: [Float] = [vel[[0, 0] as [NSNumber]].floatValue,
                          vel[[0, 1] as [NSNumber]].floatValue,
                          vel[[0, 2] as [NSNumber]].floatValue]
        let lc: [Float] = [lcov[[0, 0] as [NSNumber]].floatValue,
                           lcov[[0, 1] as [NSNumber]].floatValue,
                           lcov[[0, 2] as [NSNumber]].floatValue]
        ekf.predict()
        ekf.update(velocityMeasurement: z, logCovariance: lc)
        let ekfMs = modelMs + (CACurrentMediaTime() - t1) * 1000

        let currentRate = imuCapture.currentSampleRate

        // Update aggregate stats
        allLatencies.append(modelMs)

        inferenceInProgress = false

        DispatchQueue.main.async {
            self.inferenceTimeMs = modelMs
            self.ekfInferenceTimeMs = ekfMs
            self.sampleRate = currentRate
            self.totalInferences += 1

            // Rolling histories
            self.latencyHistory.append(modelMs)
            if self.latencyHistory.count > self.historySize {
                self.latencyHistory.removeFirst()
            }
            self.ekfLatencyHistory.append(ekfMs)
            if self.ekfLatencyHistory.count > self.historySize {
                self.ekfLatencyHistory.removeFirst()
            }
            self.sampleRateHistory.append(currentRate)
            if self.sampleRateHistory.count > self.historySize {
                self.sampleRateHistory.removeFirst()
            }

            // Aggregate stats from all latencies
            let sorted = self.allLatencies.sorted()
            self.minLatencyMs = sorted.first ?? 0
            self.maxLatencyMs = sorted.last ?? 0
            self.meanLatencyMs = sorted.reduce(0, +) / Double(sorted.count)
            self.throughputFPS = 1000.0 / self.meanLatencyMs
            let p99Idx = min(Int(Double(sorted.count) * 0.99), sorted.count - 1)
            self.p99LatencyMs = sorted[p99Idx]
            self.ekfOverheadMs = ekfMs - modelMs
        }
    }

    func exportCSV() -> URL? {
        guard !allLatencies.isEmpty else { return nil }

        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("tartanimu_perf.csv")

        var csv = "step,latency_ms\n"
        for (i, lat) in allLatencies.enumerated() {
            csv += "\(i),\(String(format: "%.3f", lat))\n"
        }

        do {
            try csv.write(to: fileURL, atomically: true, encoding: .utf8)
            return fileURL
        } catch {
            print("Failed to export CSV: \(error)")
            return nil
        }
    }
}
