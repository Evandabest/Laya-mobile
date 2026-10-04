import Darwin
import Foundation
import LayaMobile

private extension Duration {
    var milliseconds: Double {
        let parts = components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }
}

private func peakResidentBytes() -> Int64 {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    return Int64(usage.ru_maxrss)
}

private func directoryBytes(_ url: URL) -> Int64 {
    guard let enumerator = FileManager.default.enumerator(
        at: url,
        includingPropertiesForKeys: [.fileSizeKey],
        options: [.skipsHiddenFiles]
    ) else { return 0 }
    var total: Int64 = 0
    for case let fileURL as URL in enumerator {
        total += Int64((try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }
    return total
}

@main
struct LayaBenchmark {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count >= 2 else {
            FileHandle.standardError.write(
                Data("usage: LayaBenchmark MODEL_BUNDLE [WARM_ITERATIONS]\n".utf8)
            )
            Foundation.exit(2)
        }
        let bundleURL = URL(fileURLWithPath: arguments[1])
        let iterations = arguments.count >= 3 ? max(1, Int(arguments[2]) ?? 5) : 5
        let question = LayaQuestion.choice(
            .init(
                name: "department",
                instructions: "Which department should handle this email?",
                options: [
                    .init("billing", criterion: "invoices, payments, refunds"),
                    .init("technical", criterion: "bugs, outages, system errors"),
                    .init("sales", criterion: "pricing, new contracts"),
                    .init("other", criterion: "everything else"),
                ]
            )
        )
        let text = "I was charged twice. Please refund the duplicate payment."
        let clock = ContinuousClock()
        let loadStart = clock.now
        let model = try await LayaModel.load(from: bundleURL)
        let loadMilliseconds = loadStart.duration(to: clock.now).milliseconds

        let first = try model.predict(text: text, questions: [question])
        var warm: [Double] = []
        for _ in 0..<iterations {
            warm.append(try model.predict(text: text, questions: [question]).latency.milliseconds)
        }
        let report: [String: Any] = [
            "backend": LayaModel.backend,
            "model_bundle_bytes": directoryBytes(bundleURL),
            "load_ms": loadMilliseconds,
            "first_inference_ms": first.latency.milliseconds,
            "warm_iterations": iterations,
            "warm_average_ms": warm.reduce(0, +) / Double(warm.count),
            "warm_min_ms": warm.min() ?? 0,
            "warm_max_ms": warm.max() ?? 0,
            "peak_resident_bytes": peakResidentBytes(),
        ]
        let data = try JSONSerialization.data(
            withJSONObject: report,
            options: [.prettyPrinted, .sortedKeys]
        )
        print(String(decoding: data, as: UTF8.self))
    }
}
