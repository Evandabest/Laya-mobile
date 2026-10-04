import Foundation
import Testing
@testable import LayaMobile

private func modelBundleURL() -> URL? {
    if let path = ProcessInfo.processInfo.environment["LAYA_COREML_BUNDLE"] {
        return URL(fileURLWithPath: path)
    }
    let local = URL(fileURLWithPath: "/tmp/laya-ios-test2.mlpackage")
    return FileManager.default.fileExists(atPath: local.path) ? local : nil
}

@Test func realCoreMLBundleProducesReferenceDecision() async throws {
    guard let bundle = modelBundleURL() else { return }
    let model = try await LayaModel.load(from: bundle)
    let department = LayaQuestion.choice(
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
    let urgency = LayaQuestion.score(
        .init(
            name: "urgency",
            instructions: "How urgent is this request?",
            levels: ["not urgent", "soon", "critical deadline or blocking issue"]
        )
    )
    let refund = LayaQuestion.boolean(
        .init(name: "refund", instructions: "Does the customer ask for money back?")
    )
    let prediction = try model.predict(text: "", questions: [department, urgency, refund])
    #expect(prediction.results.count == 3)
    #expect(prediction.results[0].value == .choice("billing"))
    #expect(abs((prediction.results[0].probabilities["billing"] ?? 0) - 0.4002) <= 0.02)
    guard case .score(let score) = prediction.results[1].value else {
        Issue.record("Expected score result")
        return
    }
    #expect(abs(score - 1.2002) <= 0.02)
    #expect(prediction.results[2].value == .boolean(false))
    #expect(prediction.results.allSatisfy { $0.actProbability == 1 })
    #expect(prediction.backend == "Core ML FP16")
}
