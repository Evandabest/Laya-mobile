import Testing
@testable import LayaMobile

private let calibration = LayaCalibration(
    temperatures: [1, 1, 1],
    temperaturesByOptions: [:]
)

private func prepared(labels: [String], type: Int32) -> PreparedQuestion {
    .init(
        inputIDs: [],
        attentionMask: [],
        markerPositions: [],
        markerMask: [],
        questionType: type,
        optionLabels: labels,
        stateTokens: 0,
        stateTokensDropped: 0
    )
}

@Test func decodesAllTypedOutputs() throws {
    let choice = LayaQuestion.choice(
        .init(name: "department", instructions: "Route", options: [.init("billing"), .init("sales")])
    )
    let choiceResult = try LayaDecoder.decode(
        question: choice,
        prepared: prepared(labels: ["billing", "sales"], type: 0),
        logits: [4, 1],
        actionLogits: [2, 0],
        calibration: calibration
    )
    #expect(choiceResult.value == .choice("billing"))

    let score = LayaQuestion.score(
        .init(name: "urgency", instructions: "Rate", levels: ["low", "medium", "high"])
    )
    let scoreResult = try LayaDecoder.decode(
        question: score,
        prepared: prepared(labels: ["0", "1", "2"], type: 1),
        logits: [0, 0, 4],
        actionLogits: [2, 0],
        calibration: calibration
    )
    guard case .score(let scoreValue) = scoreResult.value else {
        Issue.record("Expected score result")
        return
    }
    #expect(scoreValue > 1.8)

    let boolean = LayaQuestion.boolean(.init(name: "refund", instructions: "Refund?"))
    let booleanResult = try LayaDecoder.decode(
        question: boolean,
        prepared: prepared(labels: ["false", "true"], type: 2),
        logits: [-2, 2],
        actionLogits: [2, 0],
        calibration: calibration
    )
    #expect(booleanResult.value == .boolean(true))
    #expect(booleanResult.confidence == booleanResult.answerConfidence)
}

@Test func clampsUnsafeCheckpointTemperature() {
    let value = LayaCalibration(
        temperatures: [1, 1, 1],
        temperaturesByOptions: ["choice:11+": 0.1]
    ).temperature(questionType: 0, optionCount: 12)
    #expect(value == 0.5)
}
