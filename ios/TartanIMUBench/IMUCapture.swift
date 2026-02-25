import CoreMotion
import Foundation

/// Captures IMU data at the device's maximum rate and buffers into windows.
class IMUCapture: ObservableObject {

    private let motionManager = CMMotionManager()
    private let queue = OperationQueue()

    /// Rolling buffer: windowSize samples × 6 channels (acc_xyz + gyro_xyz)
    private var buffer: [[Float]]
    private var writeIndex = 0
    private var totalSamples = 0

    let windowSize: Int
    let sampleRate: Double

    /// Callback fired each time a full window is ready
    var onWindowReady: (([[Float]]) -> Void)?

    @Published var isCapturing = false
    @Published var currentSampleRate: Double = 0

    private var rateTimer: Date?
    private var rateSampleCount = 0

    init(windowSize: Int = 200, sampleRate: Double = 100.0) {
        self.windowSize = windowSize
        self.sampleRate = sampleRate
        self.buffer = [[Float]](repeating: [Float](repeating: 0, count: 6), count: windowSize)
        queue.maxConcurrentOperationCount = 1
        queue.qualityOfService = .userInteractive
    }

    func start() {
        guard motionManager.isDeviceMotionAvailable else {
            print("Device motion not available")
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / sampleRate
        rateTimer = Date()
        rateSampleCount = 0

        motionManager.startDeviceMotionUpdates(to: queue) { [weak self] motion, error in
            guard let self, let m = motion else { return }
            self.processSample(m)
        }

        DispatchQueue.main.async {
            self.isCapturing = true
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
        DispatchQueue.main.async {
            self.isCapturing = false
        }
    }

    private func processSample(_ motion: CMDeviceMotion) {
        let acc = motion.userAcceleration
        let gyro = motion.rotationRate

        buffer[writeIndex % windowSize] = [
            Float(acc.x), Float(acc.y), Float(acc.z),
            Float(gyro.x), Float(gyro.y), Float(gyro.z)
        ]
        writeIndex += 1
        totalSamples += 1

        // Track actual sample rate
        rateSampleCount += 1
        if let timer = rateTimer, rateSampleCount >= 100 {
            let elapsed = Date().timeIntervalSince(timer)
            let rate = Double(rateSampleCount) / elapsed
            DispatchQueue.main.async {
                self.currentSampleRate = rate
            }
            rateTimer = Date()
            rateSampleCount = 0
        }

        // Fire callback every windowSize samples
        if totalSamples >= windowSize && totalSamples % windowSize == 0 {
            // Build ordered window from circular buffer
            let startIdx = writeIndex - windowSize
            var window = [[Float]]()
            window.reserveCapacity(windowSize)
            for i in 0..<windowSize {
                window.append(buffer[(startIdx + i) % windowSize])
            }
            onWindowReady?(window)
        }
    }

    /// Get the current buffer contents as an ordered array (most recent windowSize samples).
    func currentWindow() -> [[Float]]? {
        guard totalSamples >= windowSize else { return nil }
        let startIdx = writeIndex - windowSize
        var window = [[Float]]()
        window.reserveCapacity(windowSize)
        for i in 0..<windowSize {
            window.append(buffer[(startIdx + i) % windowSize])
        }
        return window
    }
}
