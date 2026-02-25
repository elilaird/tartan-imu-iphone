import CoreML
import Foundation
import QuartzCore

/// Systematic throughput benchmark across compute unit configurations.
///
/// Tests: ANE-only FP16, GPU-only FP16, CPU-only FP32, and various batch sizes.
/// Records: mean latency, p99 latency, and throughput (FPS).
class BenchmarkRunner: ObservableObject {

    struct BenchmarkConfig: Identifiable {
        let id = UUID()
        let name: String
        let computeUnits: MLComputeUnits
        let withEKF: Bool
    }

    struct BenchmarkResult: Identifiable {
        let id = UUID()
        let configName: String
        let meanLatencyMs: Double
        let p99LatencyMs: Double
        let throughputFPS: Double
        let iterations: Int
    }

    static let configs: [BenchmarkConfig] = [
        BenchmarkConfig(name: "All (ANE+GPU+CPU)", computeUnits: .all, withEKF: false),
        BenchmarkConfig(name: "All + EKF", computeUnits: .all, withEKF: true),
        BenchmarkConfig(name: "ANE+CPU (FP16)", computeUnits: .cpuAndNeuralEngine, withEKF: false),
        BenchmarkConfig(name: "ANE+CPU + EKF", computeUnits: .cpuAndNeuralEngine, withEKF: true),
        BenchmarkConfig(name: "GPU+CPU (FP16)", computeUnits: .cpuAndGPU, withEKF: false),
        BenchmarkConfig(name: "CPU Only", computeUnits: .cpuOnly, withEKF: false),
    ]

    @Published var results: [BenchmarkResult] = []
    @Published var isRunning = false
    @Published var currentConfig: String = ""
    @Published var progress: Double = 0

    private let warmupIterations = 10
    private let benchmarkIterations = 100

    func runAll() {
        guard !isRunning else { return }

        DispatchQueue.main.async {
            self.isRunning = true
            self.results = []
        }

        DispatchQueue.global(qos: .userInitiated).async { [self] in
            for (configIdx, config) in Self.configs.enumerated() {
                DispatchQueue.main.async {
                    self.currentConfig = config.name
                    self.progress = Double(configIdx) / Double(Self.configs.count)
                }

                if let result = benchmark(config: config) {
                    DispatchQueue.main.async {
                        self.results.append(result)
                    }
                }
            }

            DispatchQueue.main.async {
                self.isRunning = false
                self.progress = 1.0
                self.currentConfig = "Done"
            }
        }
    }

    private func benchmark(config: BenchmarkConfig) -> BenchmarkResult? {
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = config.computeUnits

        guard let modelURL = Bundle.main.url(forResource: "tartanimu_car", withExtension: "mlmodelc") ??
                             Bundle.main.url(forResource: "tartanimu_car", withExtension: "mlpackage"),
              let model = try? MLModel(contentsOf: modelURL, configuration: mlConfig) else {
            print("Failed to load model for config: \(config.name)")
            return nil
        }

        // Prepare inputs
        let imuInput = try! MLMultiArray(shape: [1, 6, 200], dataType: .float32)
        for i in 0..<imuInput.count { imuInput[i] = NSNumber(value: Float.random(in: -1...1)) }

        var hiddenState = try! MLMultiArray(shape: [2, 1, 512], dataType: .float32)
        var cellState = try! MLMultiArray(shape: [2, 1, 512], dataType: .float32)
        for i in 0..<hiddenState.count { hiddenState[i] = 0 }
        for i in 0..<cellState.count { cellState[i] = 0 }

        // Warmup
        for _ in 0..<warmupIterations {
            let input = try! MLDictionaryFeatureProvider(dictionary: [
                "imu_window": MLFeatureValue(multiArray: imuInput),
                "hidden": MLFeatureValue(multiArray: hiddenState),
                "cell": MLFeatureValue(multiArray: cellState),
            ])
            if let out = try? model.prediction(from: input) {
                hiddenState = out.featureValue(for: "hidden_out")!.multiArrayValue!
                cellState = out.featureValue(for: "cell_out")!.multiArrayValue!
            }
        }

        // Benchmark
        var latencies = [Double]()
        latencies.reserveCapacity(benchmarkIterations)

        let ekf: EKF? = config.withEKF ? EKF(dt: 1.0) : nil

        for _ in 0..<benchmarkIterations {
            let input = try! MLDictionaryFeatureProvider(dictionary: [
                "imu_window": MLFeatureValue(multiArray: imuInput),
                "hidden": MLFeatureValue(multiArray: hiddenState),
                "cell": MLFeatureValue(multiArray: cellState),
            ])

            let start = CACurrentMediaTime()
            if let out = try? model.prediction(from: input) {
                // Run EKF predict + update if enabled (included in timing)
                if let ekf {
                    let vel = out.featureValue(for: "velocity")!.multiArrayValue!
                    let lcov = out.featureValue(for: "log_covariance")!.multiArrayValue!
                    let z: [Float] = [vel[[0, 0] as [NSNumber]].floatValue,
                                      vel[[0, 1] as [NSNumber]].floatValue,
                                      vel[[0, 2] as [NSNumber]].floatValue]
                    let lc: [Float] = [lcov[[0, 0] as [NSNumber]].floatValue,
                                       lcov[[0, 1] as [NSNumber]].floatValue,
                                       lcov[[0, 2] as [NSNumber]].floatValue]
                    ekf.predict()
                    ekf.update(velocityMeasurement: z, logCovariance: lc)
                }

                let elapsed = (CACurrentMediaTime() - start) * 1000
                latencies.append(elapsed)
                hiddenState = out.featureValue(for: "hidden_out")!.multiArrayValue!
                cellState = out.featureValue(for: "cell_out")!.multiArrayValue!
            }
        }

        guard !latencies.isEmpty else { return nil }

        let sorted = latencies.sorted()
        let mean = sorted.reduce(0, +) / Double(sorted.count)
        let p99Idx = min(Int(Double(sorted.count) * 0.99), sorted.count - 1)
        let p99 = sorted[p99Idx]
        let fps = 1000.0 / mean

        return BenchmarkResult(
            configName: config.name,
            meanLatencyMs: mean,
            p99LatencyMs: p99,
            throughputFPS: fps,
            iterations: latencies.count
        )
    }
}
