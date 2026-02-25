import SwiftUI
import Charts

// MARK: - Main Dashboard (Performance-focused)

struct TartanIMUDashboard: View {
    @StateObject private var runner = TartanIMURunner()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {

                    // Top metrics bar
                    HStack(spacing: 12) {
                        MetricTile(
                            label: "Model",
                            value: String(format: "%.2f ms", runner.inferenceTimeMs),
                            color: runner.inferenceTimeMs < 5 ? .green : .orange
                        )
                        MetricTile(
                            label: "+EKF",
                            value: String(format: "%.2f ms", runner.ekfInferenceTimeMs),
                            color: .orange
                        )
                        MetricTile(
                            label: "FPS",
                            value: String(format: "%.0f", runner.throughputFPS),
                            color: .blue
                        )
                        MetricTile(
                            label: "IMU Hz",
                            value: String(format: "%.0f", runner.sampleRate),
                            color: .cyan
                        )
                        MetricTile(
                            label: "Calls",
                            value: "\(runner.totalInferences)",
                            color: .primary
                        )
                    }
                    .padding(.horizontal)

                    // Latency time-series: model vs model+EKF
                    if !runner.latencyHistory.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Inference Latency")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .padding(.horizontal)

                            Chart {
                                ForEach(runner.latencyHistory.indices, id: \.self) { i in
                                    LineMark(
                                        x: .value("Step", i),
                                        y: .value("ms", runner.latencyHistory[i]),
                                        series: .value("Source", "Model")
                                    )
                                    .foregroundStyle(.blue)
                                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                                }
                                ForEach(runner.ekfLatencyHistory.indices, id: \.self) { i in
                                    LineMark(
                                        x: .value("Step", i),
                                        y: .value("ms", runner.ekfLatencyHistory[i]),
                                        series: .value("Source", "Model + EKF")
                                    )
                                    .foregroundStyle(.orange)
                                    .lineStyle(StrokeStyle(lineWidth: 1.5))
                                }
                            }
                            .chartXAxisLabel("Step")
                            .chartYAxisLabel("Latency (ms)")
                            .chartForegroundStyleScale([
                                "Model": Color.blue,
                                "Model + EKF": Color.orange,
                            ])
                            .frame(height: 180)
                            .padding(.horizontal)
                        }
                    }

                    // Latency distribution histogram
                    if runner.latencyHistory.count > 10 {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Latency Distribution")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .padding(.horizontal)

                            LatencyHistogramView(latencies: runner.latencyHistory)
                                .frame(height: 140)
                                .padding(.horizontal)
                        }
                    }

                    // IMU sample rate over time
                    if !runner.sampleRateHistory.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("IMU Sample Rate")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .padding(.horizontal)

                            Chart(runner.sampleRateHistory.indices, id: \.self) { i in
                                LineMark(
                                    x: .value("Step", i),
                                    y: .value("Hz", runner.sampleRateHistory[i])
                                )
                                .foregroundStyle(.cyan)
                                .lineStyle(StrokeStyle(lineWidth: 1.5))

                                // Target line at 100 Hz
                                RuleMark(y: .value("Target", 100))
                                    .foregroundStyle(.secondary.opacity(0.5))
                                    .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
                            }
                            .chartXAxisLabel("Step")
                            .chartYAxisLabel("Hz")
                            .frame(height: 120)
                            .padding(.horizontal)
                        }
                    }

                    // Stats summary table
                    if runner.totalInferences > 0 {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Session Statistics")
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .padding(.horizontal)

                            VStack(spacing: 0) {
                                StatRow(label: "Mean latency", value: String(format: "%.2f ms", runner.meanLatencyMs))
                                StatRow(label: "Min latency", value: String(format: "%.2f ms", runner.minLatencyMs))
                                StatRow(label: "Max latency", value: String(format: "%.2f ms", runner.maxLatencyMs))
                                StatRow(label: "P99 latency", value: String(format: "%.2f ms", runner.p99LatencyMs))
                                StatRow(label: "EKF overhead", value: String(format: "%.3f ms", runner.ekfOverheadMs))
                                StatRow(label: "Throughput", value: String(format: "%.0f FPS", runner.throughputFPS))
                                StatRow(label: "Total inferences", value: "\(runner.totalInferences)")
                                StatRow(label: "Dropped windows", value: "\(runner.droppedWindows)",
                                        valueColor: runner.droppedWindows > 0 ? .red : .primary)
                            }
                            .background(Color(.systemGroupedBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 12))
                            .padding(.horizontal)
                        }
                    }

                    // Controls
                    HStack(spacing: 16) {
                        if runner.isRunning {
                            Button("Stop") { runner.stop() }
                                .buttonStyle(.borderedProminent)
                                .tint(.red)
                        } else {
                            Button("Start") { runner.start() }
                                .buttonStyle(.borderedProminent)
                        }

                        Button("Reset") { runner.reset() }
                            .buttonStyle(.bordered)

                        ShareLink(item: runner.exportCSV() ?? URL(fileURLWithPath: "/dev/null")) {
                            Label("Export", systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.bordered)
                        .disabled(runner.totalInferences == 0)
                    }
                    .padding(.bottom)
                }
            }
            .navigationTitle("Live Performance")
        }
    }
}

// MARK: - Metric Tile

struct MetricTile: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.bold)
                .foregroundStyle(color)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Stat Row

struct StatRow: View {
    let label: String
    let value: String
    var valueColor: Color = .primary

    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .font(.system(.subheadline, design: .monospaced))
                .fontWeight(.medium)
                .foregroundStyle(valueColor)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
    }
}

// MARK: - Latency Histogram

struct LatencyHistogramView: View {
    let latencies: [Double]

    var body: some View {
        let bins = computeBins()
        Chart(bins, id: \.lowerBound) { bin in
            BarMark(
                x: .value("Latency (ms)", String(format: "%.1f", bin.lowerBound)),
                y: .value("Count", bin.count)
            )
            .foregroundStyle(.blue.opacity(0.7))
        }
        .chartXAxisLabel("Latency (ms)")
        .chartYAxisLabel("Count")
    }

    private func computeBins() -> [HistBin] {
        guard let lo = latencies.min(), let hi = latencies.max(), hi > lo else {
            return []
        }

        let nBins = 15
        let step = (hi - lo) / Double(nBins)
        var bins = (0..<nBins).map { i in
            HistBin(lowerBound: lo + Double(i) * step, count: 0)
        }

        for v in latencies {
            var idx = Int((v - lo) / step)
            idx = min(idx, nBins - 1)
            bins[idx].count += 1
        }

        return bins
    }
}

struct HistBin {
    let lowerBound: Double
    var count: Int
}

// MARK: - Preview

#Preview {
    TartanIMUDashboard()
}
