import CoreML
import Foundation
import QuartzCore
import simd

/// Runs TartanIMU CoreML inference with persistent LSTM state.
class TartanIMURunner: ObservableObject {

    private var model: MLModel?
    private let imuCapture: IMUCapture

    // LSTM state — persisted across calls
    private var hiddenState: MLMultiArray
    private var cellState: MLMultiArray

    // Performance tracking
    @Published var inferenceTimeMs: Double = 0
    @Published var throughputFPS: Double = 0
    @Published var velocityEstimate: SIMD3<Float> = .zero
    @Published var trajectoryPoints: [SIMD2<Float>] = []
    @Published var velocityHistory: [Float] = []
    @Published var isRunning = false
    @Published var sampleRate: Double = 0

    private var inferenceTimings = [Double]()
    private var integratedPos = SIMD3<Float>.zero

    init() {
        // Initialize LSTM hidden states to zero
        hiddenState = try! MLMultiArray(shape: [2, 1, 512], dataType: .float32)
        cellState = try! MLMultiArray(shape: [2, 1, 512], dataType: .float32)
        for i in 0..<hiddenState.count { hiddenState[i] = 0 }
        for i in 0..<cellState.count { cellState[i] = 0 }

        imuCapture = IMUCapture(windowSize: 200, sampleRate: 100.0)

        loadModel()

        // Run inference when a full window is ready
        imuCapture.onWindowReady = { [weak self] window in
            self?.runInference(window: window)
        }
    }

    private func loadModel() {
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Allow ANE + GPU + CPU

        // Try to load the compiled model from the app bundle
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
        DispatchQueue.main.async {
            self.isRunning = true
        }
    }

    func stop() {
        imuCapture.stop()
        DispatchQueue.main.async {
            self.isRunning = false
        }
    }

    func reset() {
        for i in 0..<hiddenState.count { hiddenState[i] = 0 }
        for i in 0..<cellState.count { cellState[i] = 0 }
        integratedPos = .zero
        DispatchQueue.main.async {
            self.trajectoryPoints = []
            self.velocityHistory = []
            self.inferenceTimings = []
            self.velocityEstimate = .zero
            self.inferenceTimeMs = 0
            self.throughputFPS = 0
        }
    }

    private func runInference(window: [[Float]]) {
        guard let model else { return }

        let startTime = CACurrentMediaTime()

        // Pack buffer into MLMultiArray [1, 6, 200]
        let imuInput = try! MLMultiArray(shape: [1, 6, 200], dataType: .float32)
        for t in 0..<200 {
            let sample = window[t]
            for ch in 0..<6 {
                imuInput[[0, ch, t] as [NSNumber]] = NSNumber(value: sample[ch])
            }
        }

        // Build prediction input
        let inputFeatures = try! MLDictionaryFeatureProvider(dictionary: [
            "imu_window": MLFeatureValue(multiArray: imuInput),
            "hidden": MLFeatureValue(multiArray: hiddenState),
            "cell": MLFeatureValue(multiArray: cellState),
        ])

        guard let output = try? model.prediction(from: inputFeatures) else {
            print("Inference failed")
            return
        }

        // Update persistent LSTM state
        hiddenState = output.featureValue(for: "hidden_out")!.multiArrayValue!
        cellState = output.featureValue(for: "cell_out")!.multiArrayValue!

        // Parse velocity output
        let velocity = output.featureValue(for: "velocity")!.multiArrayValue!
        let vx = velocity[[0, 0] as [NSNumber]].floatValue
        let vy = velocity[[0, 1] as [NSNumber]].floatValue
        let vz = velocity[[0, 2] as [NSNumber]].floatValue

        let elapsed = (CACurrentMediaTime() - startTime) * 1000  // ms

        // Dead-reckoning integration (1 Hz steps, 1 second per window)
        integratedPos.x += vx * 1.0
        integratedPos.y += vy * 1.0

        let currentRate = imuCapture.currentSampleRate

        DispatchQueue.main.async {
            self.inferenceTimeMs = elapsed
            self.velocityEstimate = SIMD3(vx, vy, vz)
            self.trajectoryPoints.append(SIMD2(self.integratedPos.x, self.integratedPos.y))
            self.sampleRate = currentRate

            self.velocityHistory.append(vx)
            if self.velocityHistory.count > 120 {
                self.velocityHistory.removeFirst()
            }

            self.inferenceTimings.append(elapsed)
            if self.inferenceTimings.count > 100 {
                self.inferenceTimings.removeFirst()
            }
            let avgMs = self.inferenceTimings.reduce(0, +) / Double(self.inferenceTimings.count)
            self.throughputFPS = 1000.0 / avgMs
        }
    }

    func exportCSV() -> URL? {
        guard !trajectoryPoints.isEmpty else { return nil }

        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("tartanimu_trajectory.csv")

        var csv = "step,x,y\n"
        for (i, pt) in trajectoryPoints.enumerated() {
            csv += "\(i),\(pt.x),\(pt.y)\n"
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
