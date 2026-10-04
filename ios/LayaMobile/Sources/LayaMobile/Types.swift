import Foundation

public struct LayaStateField: Equatable, Sendable {
    public let key: String
    public let value: LayaJSONValue

    public init(_ key: String, _ value: LayaJSONValue) {
        self.key = key
        self.value = value
    }
}

public indirect enum LayaJSONValue: Equatable, Sendable {
    case string(String)
    case integer(Int)
    case number(Double)
    case boolean(Bool)
    case object([LayaStateField])
    case array([LayaJSONValue])
    case null
}

public enum LayaState: Equatable, Sendable {
    case text(String)
    case object([LayaStateField])
    case conversation([LayaJSONValue])

    public func serialized() throws -> String {
        switch self {
        case .text(let value):
            value
        case .object(let fields):
            try LayaJSONValue.object(fields).serialized()
        case .conversation(let turns):
            try LayaJSONValue.array(turns).serialized()
        }
    }

    var truncatesFromLeft: Bool {
        if case .conversation = self { return true }
        return false
    }
}

private extension LayaJSONValue {
    func serialized() throws -> String {
        switch self {
        case .string(let value):
            return Self.quote(value)
        case .integer(let value):
            return String(value)
        case .number(let value):
            guard value.isFinite else {
                throw LayaError.unsupportedState("JSON numbers must be finite")
            }
            return String(value)
        case .boolean(let value):
            return value ? "true" : "false"
        case .object(let fields):
            var seen: Set<String> = []
            let members = try fields.map { field in
                guard seen.insert(field.key).inserted else {
                    throw LayaError.unsupportedState(
                        "JSON objects must not contain duplicate key '\(field.key)'"
                    )
                }
                return "\(Self.quote(field.key)): \(try field.value.serialized())"
            }
            return "{\(members.joined(separator: ", "))}"
        case .array(let values):
            return "[\(try values.map { try $0.serialized() }.joined(separator: ", "))]"
        case .null:
            return "null"
        }
    }

    static func quote(_ value: String) -> String {
        var result = "\""
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 0x08: result += "\\b"
            case 0x09: result += "\\t"
            case 0x0A: result += "\\n"
            case 0x0C: result += "\\f"
            case 0x0D: result += "\\r"
            case 0x22: result += "\\\""
            case 0x5C: result += "\\\\"
            case 0x00...0x1F:
                result += String(format: "\\u%04x", scalar.value)
            default:
                result.unicodeScalars.append(scalar)
            }
        }
        return result + "\""
    }
}

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
