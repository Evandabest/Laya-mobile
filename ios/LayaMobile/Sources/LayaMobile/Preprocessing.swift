import Foundation

public struct PreparedQuestion: Equatable, Sendable {
    public let inputIDs: [Int32]
    public let attentionMask: [Int32]
    public let markerPositions: [Int32]
    public let markerMask: [Int32]
    public let questionType: Int32
    public let optionLabels: [String]
    public let stateTokens: Int
    public let stateTokensDropped: Int
}

public struct LayaPreprocessor: Sendable {
    public static let sequenceLength = 512
    public static let markerSlots = 48
    public static let headMaxLength = 192

    private let tokenizer: LayaTokenizer

    public init(tokenizer: LayaTokenizer) {
        self.tokenizer = tokenizer
    }

    public func prepare(text: String, question: LayaQuestion) throws -> PreparedQuestion {
        try prepare(state: .text(text), question: question)
    }

    public func prepare(state: LayaState, question: LayaQuestion) throws -> PreparedQuestion {
        let rendered = try render(question)
        let instructions = rendered.instructions.replacingOccurrences(
            of: tokenizer.maskToken,
            with: " "
        )
        var headIDs = tokenizer.encode("\(rendered.typeName) question: \(instructions)")
        var optionIDs = rendered.options.map { option in
            [tokenizer.maskTokenID]
                + Array(
                    tokenizer.encode(
                        " " + option.replacingOccurrences(of: tokenizer.maskToken, with: " ")
                    ).prefix(48)
                )
        }
        var optionBudget = Self.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        if optionBudget < 16 {
            let perOption = max(4, (Self.headMaxLength - 16) / max(1, optionIDs.count))
            optionIDs = optionIDs.map { Array($0.prefix(perOption)) }
            optionBudget = Self.headMaxLength - optionIDs.reduce(0) { $0 + $1.count }
        }
        headIDs = Array(headIDs.prefix(max(8, optionBudget)))

        var ids = [tokenizer.clsTokenID] + headIDs + [tokenizer.sepTokenID]
        var markers: [Int] = []
        for option in optionIDs {
            markers.append(ids.count)
            ids.append(contentsOf: option)
        }
        ids.append(tokenizer.sepTokenID)

        let stateText = try state.serialized()
        let stateIDs = tokenizer.encode(
            stateText.replacingOccurrences(of: tokenizer.maskToken, with: " ")
        )
        let room = max(0, Self.sequenceLength - ids.count - 1)
        let keptState = state.truncatesFromLeft
            ? Array(stateIDs.suffix(room))
            : Array(stateIDs.prefix(room))
        ids.append(contentsOf: keptState)
        ids.append(tokenizer.sepTokenID)
        ids = Array(ids.prefix(Self.sequenceLength))
        markers = markers.filter { $0 < Self.sequenceLength }

        var paddedIDs = ids.map(Int32.init)
        paddedIDs += Array(
            repeating: Int32(tokenizer.padTokenID),
            count: Self.sequenceLength - paddedIDs.count
        )
        var attention = Array(repeating: Int32(0), count: Self.sequenceLength)
        attention.replaceSubrange(0..<ids.count, with: repeatElement(Int32(1), count: ids.count))
        var markerPositions = markers.map(Int32.init)
        markerPositions += Array(
            repeating: 0,
            count: Self.markerSlots - markerPositions.count
        )
        let markerMask = Array(repeating: Int32(1), count: markers.count)
            + Array(repeating: Int32(0), count: Self.markerSlots - markers.count)
        return PreparedQuestion(
            inputIDs: paddedIDs,
            attentionMask: attention,
            markerPositions: markerPositions,
            markerMask: markerMask,
            questionType: rendered.questionType,
            optionLabels: rendered.labels,
            stateTokens: stateIDs.count,
            stateTokensDropped: stateIDs.count - keptState.count
        )
    }

    private func render(_ question: LayaQuestion) throws -> RenderedQuestion {
        switch question {
        case .choice(let value):
            guard !value.name.isEmpty, !value.instructions.isEmpty else {
                throw LayaError.invalidQuestion("Choice name and instructions must not be empty")
            }
            try validateOptionCount(value.options.count)
            let rendered = value.options.map { option in
                if let criterion = option.criterion, !criterion.isEmpty {
                    return "\(option.label): \(criterion)"
                }
                return option.label
            }
            return .init(
                typeName: "choice",
                questionType: 0,
                instructions: value.instructions,
                options: rendered,
                labels: value.options.map(\.label)
            )
        case .score(let value):
            guard !value.name.isEmpty, !value.instructions.isEmpty else {
                throw LayaError.invalidQuestion("Score name and instructions must not be empty")
            }
            try validateOptionCount(value.levels.count)
            return .init(
                typeName: "score",
                questionType: 1,
                instructions: value.instructions,
                options: value.levels.enumerated().map { "level \($0.offset): \($0.element)" },
                labels: value.levels.indices.map(String.init)
            )
        case .boolean(let value):
            guard !value.name.isEmpty, !value.instructions.isEmpty else {
                throw LayaError.invalidQuestion("Boolean name and instructions must not be empty")
            }
            let falseLabel = value.falseLabel.trimmingCharacters(in: .whitespacesAndNewlines)
            let trueLabel = value.trueLabel.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !falseLabel.isEmpty, !trueLabel.isEmpty, falseLabel != trueLabel else {
                throw LayaError.invalidQuestion("Boolean labels must be non-empty and distinct")
            }
            return .init(
                typeName: "noul",
                questionType: 2,
                instructions: value.instructions,
                options: [
                    "\(falseLabel): \(value.falseCriterion ?? "no, the statement does not hold")",
                    "\(trueLabel): \(value.trueCriterion ?? "yes, the statement holds")",
                ],
                labels: [falseLabel, trueLabel]
            )
        }
    }

    private func validateOptionCount(_ count: Int) throws {
        guard (1...Self.markerSlots).contains(count) else {
            throw LayaError.invalidQuestion("Questions require between 1 and 48 options")
        }
    }
}

private struct RenderedQuestion {
    let typeName: String
    let questionType: Int32
    let instructions: String
    let options: [String]
    let labels: [String]
}
