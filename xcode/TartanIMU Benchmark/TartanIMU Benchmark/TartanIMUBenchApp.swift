import SwiftUI

@main
struct TartanIMUBenchApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                TartanIMUDashboard()
                    .tabItem {
                        Label("Live", systemImage: "location.fill")
                    }

                BenchmarkView()
                    .tabItem {
                        Label("Benchmark", systemImage: "gauge.with.needle")
                    }
            }
        }
    }
}

// MARK: - Benchmark View

struct BenchmarkView: View {
    @StateObject private var runner = BenchmarkRunner()

    var body: some View {
        NavigationStack {
            List {
                if runner.isRunning {
                    Section {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Running: \(runner.currentConfig)")
                                .font(.headline)
                            ProgressView(value: runner.progress)
                        }
                        .padding(.vertical, 4)
                    }
                }

                if !runner.results.isEmpty {
                    Section("Results") {
                        ForEach(runner.results) { result in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(result.configName)
                                    .font(.headline)
                                HStack(spacing: 16) {
                                    VStack(alignment: .leading) {
                                        Text("Mean")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(String(format: "%.2f ms", result.meanLatencyMs))
                                            .font(.system(.body, design: .monospaced))
                                    }
                                    VStack(alignment: .leading) {
                                        Text("P99")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(String(format: "%.2f ms", result.p99LatencyMs))
                                            .font(.system(.body, design: .monospaced))
                                    }
                                    VStack(alignment: .leading) {
                                        Text("FPS")
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                        Text(String(format: "%.0f", result.throughputFPS))
                                            .font(.system(.body, design: .monospaced))
                                            .foregroundStyle(.blue)
                                    }
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }

                Section {
                    Button(runner.isRunning ? "Running..." : "Run Benchmark") {
                        runner.runAll()
                    }
                    .disabled(runner.isRunning)
                }
            }
            .navigationTitle("Benchmark")
        }
    }
}
