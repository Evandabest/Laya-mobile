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

@Test func structuredStatesUseUpstreamJSONRendering() throws {
    let object = LayaState.object([
        .init("message", .string("Charged \"twice\" 👋")),
        .init("attempt", .integer(2)),
        .init("paid", .boolean(true)),
        .init("metadata", .object([
            .init("source", .string("iOS")),
            .init("missing", .null),
        ])),
    ])
    #expect(
        try object.serialized()
            == #"{"message": "Charged \"twice\" 👋", "attempt": 2, "paid": true, "metadata": {"source": "iOS", "missing": null}}"#
    )

    let conversation = LayaState.conversation([
        .object([.init("role", .string("user")), .init("content", .string("hello"))]),
        .object([.init("role", .string("assistant")), .init("content", .string("hi"))]),
    ])
    #expect(
        try conversation.serialized()
            == #"[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]"#
    )
}

@Test func structuredStateRejectsInvalidJSON() {
    #expect(throws: LayaError.self) {
        try LayaState.object([.init("score", .number(.infinity))]).serialized()
    }
    #expect(throws: LayaError.self) {
        try LayaState.object([.init("same", .null), .init("same", .null)]).serialized()
    }
}
