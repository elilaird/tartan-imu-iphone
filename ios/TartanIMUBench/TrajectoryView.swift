import SwiftUI
import Charts

// MARK: - Main Dashboard

struct TartanIMUDashboard: View {
    @StateObject private var runner = TartanIMURunner()

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {

                // Metrics bar
                HStack(spacing: 24) {
                    MetricTile(
                        label: "Inference",
                        value: String(format: "%.1f ms", runner.inferenceTimeMs),
                        color: runner.inferenceTimeMs < 5 ? .green : .orange
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
                        label: "Vx",
                        value: String(format: "%.2f m/s", runner.velocityEstimate.x),
                        color: .purple
                    )
                }
                .padding(.horizontal)

                // 2D Trajectory Canvas
                TrajectoryCanvas(points: runner.trajectoryPoints)
                    .frame(maxWidth: .infinity)
                    .frame(height: 300)
                    .background(Color(.systemGroupedBackground))
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    .padding(.horizontal)

                // Velocity time-series chart
                if !runner.velocityHistory.isEmpty {
                    Chart(runner.velocityHistory.indices, id: \.self) { i in
                        LineMark(
                            x: .value("t", i),
                            y: .value("Vx", runner.velocityHistory[i])
                        )
                        .foregroundStyle(.blue)
                    }
                    .frame(height: 120)
                    .padding(.horizontal)
                }

                Spacer()

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
                    .disabled(runner.trajectoryPoints.isEmpty)
                }
                .padding(.bottom)
            }
            .navigationTitle("TartanIMU Live")
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
                .font(.system(.title3, design: .monospaced))
                .fontWeight(.bold)
                .foregroundStyle(color)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

// MARK: - Trajectory Canvas

struct TrajectoryCanvas: View {
    let points: [SIMD2<Float>]

    var body: some View {
        Canvas { ctx, size in
            guard points.count > 1 else {
                // Draw placeholder text
                ctx.draw(
                    Text("Waiting for data...")
                        .font(.caption)
                        .foregroundColor(.secondary),
                    at: CGPoint(x: size.width / 2, y: size.height / 2)
                )
                return
            }

            let margin: CGFloat = 20
            let drawSize = CGSize(
                width: size.width - 2 * margin,
                height: size.height - 2 * margin
            )

            // Auto-scale to fit all points
            let xs = points.map(\.x), ys = points.map(\.y)
            let xMin = xs.min()!, xMax = xs.max()!
            let yMin = ys.min()!, yMax = ys.max()!
            let xRange = max(xMax - xMin, 0.001)
            let yRange = max(yMax - yMin, 0.001)

            // Uniform scaling to preserve aspect ratio
            let scale = min(Float(drawSize.width) / xRange, Float(drawSize.height) / yRange)

            func toScreen(_ p: SIMD2<Float>) -> CGPoint {
                CGPoint(
                    x: margin + CGFloat((p.x - xMin) * scale),
                    y: size.height - margin - CGFloat((p.y - yMin) * scale)
                )
            }

            // Draw path
            var path = Path()
            path.move(to: toScreen(points[0]))
            for p in points.dropFirst() {
                path.addLine(to: toScreen(p))
            }
            ctx.stroke(path, with: .color(.blue), lineWidth: 2)

            // Start marker (green)
            let startPt = toScreen(points.first!)
            ctx.fill(
                Circle().path(in: CGRect(
                    x: startPt.x - 5, y: startPt.y - 5,
                    width: 10, height: 10
                )),
                with: .color(.green)
            )

            // Current position marker (red)
            let endPt = toScreen(points.last!)
            ctx.fill(
                Circle().path(in: CGRect(
                    x: endPt.x - 5, y: endPt.y - 5,
                    width: 10, height: 10
                )),
                with: .color(.red)
            )
        }
    }
}

// MARK: - Preview

#Preview {
    TartanIMUDashboard()
}
