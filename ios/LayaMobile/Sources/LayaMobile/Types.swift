import Foundation

public struct ChoiceOption: Equatable, Sendable {
    public let label: String
    public let criterion: String?

    public init(_ label: String, criterion: String? = nil) {
        self.label = label
        self.criterion = criterion
    }
}

public struct ChoiceQuestion: Equatable, Sendable {
    public let name: String
    public let instructions: String
    public let options: [ChoiceOption]

    public init(name: String, instructions: String, options: [ChoiceOption]) {
        self.name = name
        self.instructions = instructions
        self.options = options
    }
}

public struct ScoreQuestion: Equatable, Sendable {
    public let name: String
    public let instructions: String
    public let levels: [String]

    public init(name: String, instructions: String, levels: [String]) {
        self.name = name
        self.instructions = instructions
        self.levels = levels
    }
}

public struct BooleanQuestion: Equatable, Sendable {
    public let name: String
    public let instructions: String
    public let falseCriterion: String?
    public let trueCriterion: String?
    public let falseLabel: String
    public let trueLabel: String

    public init(
        name: String,
        instructions: String,
        falseCriterion: String? = nil,
        trueCriterion: String? = nil,
        falseLabel: String = "false",
        trueLabel: String = "true"
    ) {
        self.name = name
        self.instructions = instructions
        self.falseCriterion = falseCriterion
        self.trueCriterion = trueCriterion
        self.falseLabel = falseLabel
        self.trueLabel = trueLabel
    }
}

public enum LayaQuestion: Equatable, Sendable {
    case choice(ChoiceQuestion)
    case score(ScoreQuestion)
    case boolean(BooleanQuestion)

    public var name: String {
        switch self {
        case .choice(let question): question.name
        case .score(let question): question.name
        case .boolean(let question): question.name
        }
    }
}

public struct DecisionResult: Equatable, Sendable {
    public enum Value: Equatable, Sendable {
        case choice(String)
        case score(Double)
        case boolean(Bool)
    }

    public let questionName: String
    public let value: Value
    public let confidence: Double
    public let answerConfidence: Double
    public let probabilities: [String: Double]
    public let actProbability: Double

    public init(
        questionName: String,
        value: Value,
        confidence: Double,
        answerConfidence: Double,
        probabilities: [String: Double],
        actProbability: Double
    ) {
        self.questionName = questionName
        self.value = value
        self.confidence = confidence
        self.answerConfidence = answerConfidence
        self.probabilities = probabilities
        self.actProbability = actProbability
    }
}

public struct PredictionUsage: Equatable, Sendable {
    public let inputTokens: Int
    public let stateTokens: Int
    public let stateTokensDropped: Int

    public var truncated: Bool { stateTokensDropped > 0 }
}

public struct LayaPrediction: Equatable, Sendable {
    public let results: [DecisionResult]
    public let usage: PredictionUsage
    public let latency: Duration
    public let backend: String
}
