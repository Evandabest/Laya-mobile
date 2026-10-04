import Foundation
import Testing
@testable import LayaMobile

private func parityRepositoryRoot() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func parityModelBundleURL() -> URL? {
    if let path = ProcessInfo.processInfo.environment["LAYA_COREML_BUNDLE"] {
        return URL(fileURLWithPath: path)
    }
    let local = URL(fileURLWithPath: "/tmp/laya-ios-test2.mlpackage")
    return FileManager.default.fileExists(atPath: local.path) ? local : nil
}

private func paritySoftmax(_ values: [Double]) -> [Double] {
    let maximum = values.max() ?? 0
    let exponentials = values.map { exp($0 - maximum) }
    let total = exponentials.reduce(0, +)
    return exponentials.map { $0 / total }
}

private func recoveredTemperature(logits: [Double], probabilities: [Double], count: Int) -> Double? {
    for left in 0..<count {
        for right in (left + 1)..<count
        where abs(logits[left] - logits[right]) > 1e-8
            && probabilities[left] > 0
            && probabilities[right] > 0
        {
            return (logits[left] - logits[right])
                / log(probabilities[left] / probabilities[right])
        }
    }
    return nil
}

@Test func allCoreMLFixtureRowsMatchThroughSwiftRuntime() async throws {
    guard let modelBundle = parityModelBundleURL() else { return }
    let fixtureURL = parityRepositoryRoot().appending(
        path: "parity_tests/fixtures/laya-ios-english-v1.json"
    )
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL))
    let bundle = try #require(object as? [String: Any])
    let cases = try #require(bundle["cases"] as? [[String: Any]])
    let model = try await LayaModel.load(from: modelBundle)
    var checked = 0
    var maximumProbabilityError = 0.0

    for fixtureCase in cases {
        let rows = try #require(fixtureCase["rows"] as? [[String: Any]])
        for row in rows {
            let optionCount = try #require(row["option_count"] as? Int)
            let expectedLogits = (try #require(row["logits"] as? [Double]))
            let expectedProbabilities = try #require(row["probabilities"] as? [Double])
            let prepared = PreparedQuestion(
                inputIDs: (try #require(row["input_ids"] as? [Int])).map(Int32.init),
                attentionMask: (try #require(row["attention_mask"] as? [Bool])).map { $0 ? 1 : 0 },
                markerPositions: (try #require(row["marker_pos"] as? [Int])).map(Int32.init),
                markerMask: (try #require(row["marker_mask"] as? [Bool])).map { $0 ? 1 : 0 },
                questionType: Int32(try #require(row["qtype"] as? Int)),
                optionLabels: (0..<optionCount).map(String.init),
                stateTokens: 0,
                stateTokensDropped: 0
            )
            let actual = try model.prediction(prepared)
            let temperature = optionCount == 1
                ? 1
                : try #require(
                    recoveredTemperature(
                        logits: expectedLogits,
                        probabilities: expectedProbabilities,
                        count: optionCount
                    )
                )
            let probabilities = paritySoftmax(
                Array(actual.logits.prefix(optionCount)).map { $0 / temperature }
            )
            let probabilityError = zip(probabilities, expectedProbabilities)
                .map { abs($0 - $1) }
                .max() ?? 0
            maximumProbabilityError = max(maximumProbabilityError, probabilityError)
            let actualDecision = probabilities.indices.max {
                probabilities[$0] < probabilities[$1]
            }
            let expectedDecision = expectedProbabilities.indices.max {
                expectedProbabilities[$0] < expectedProbabilities[$1]
            }
            #expect(actualDecision == expectedDecision)
            #expect(probabilityError <= 0.02)
            checked += 1
        }
    }
    #expect(checked == 44)
    #expect(maximumProbabilityError <= 0.02)
}
