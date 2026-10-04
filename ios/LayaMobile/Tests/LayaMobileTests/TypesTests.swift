import Testing
@testable import LayaMobile

@Test func typedQuestionsExposeStableNames() {
    let questions: [LayaQuestion] = [
        .choice(.init(name: "department", instructions: "Route it", options: [.init("billing")])),
        .score(.init(name: "urgency", instructions: "Rate it", levels: ["low", "high"])),
        .boolean(.init(name: "refund", instructions: "Is a refund requested?")),
    ]

    #expect(questions.map(\.name) == ["department", "urgency", "refund"])
}

@Test func usageReportsTruncation() {
    let usage = PredictionUsage(inputTokens: 512, stateTokens: 600, stateTokensDropped: 100)
    #expect(usage.truncated)
}
