import Foundation

struct LayaCalibration: Sendable {
    let temperatures: [Double]
    let temperaturesByOptions: [String: Double]

    func temperature(questionType: Int32, optionCount: Int) -> Double {
        let typeNames = ["choice", "score", "noul"]
        let typeIndex = Int(questionType)
        let size = switch optionCount {
        case ...2: "2"
        case 3...5: "3-5"
        case 6...10: "6-10"
        default: "11+"
        }
        let fallback = temperatures.indices.contains(typeIndex) ? temperatures[typeIndex] : 1
        let key = "\(typeNames[typeIndex]):\(size)"
        return Self.clamp(temperaturesByOptions[key] ?? fallback)
    }

    private static func clamp(_ value: Double) -> Double {
        value.isFinite ? min(5, max(0.5, value)) : 1
    }
}

enum LayaDecoder {
    static func decode(
        question: LayaQuestion,
        prepared: PreparedQuestion,
        logits: [Double],
        actionLogits: [Double],
        calibration: LayaCalibration
    ) throws -> DecisionResult {
        let count = prepared.optionLabels.count
        guard logits.count >= count, actionLogits.count >= 2 else {
            throw LayaError.model("Core ML returned tensors with unexpected shapes")
        }
        let temperature = calibration.temperature(
            questionType: prepared.questionType,
            optionCount: count
        )
        let probabilities = softmax(Array(logits.prefix(count)).map { $0 / temperature })
        let action = softmax(actionLogits)
        let topIndex = probabilities.indices.max { probabilities[$0] < probabilities[$1] } ?? 0
        let answerConfidence = probabilities[topIndex]
        let entropyConfidence: Double
        if count < 2 {
            entropyConfidence = 1
        } else {
            let entropy = -probabilities.reduce(0) { partial, probability in
                partial + probability * log(max(1e-12, probability))
            }
            entropyConfidence = min(1, max(0, 1 - entropy / log(Double(count))))
        }
        let probabilityMap = Dictionary(
            uniqueKeysWithValues: zip(prepared.optionLabels, probabilities).map {
                ($0.0, rounded($0.1))
            }
        )
        let value: DecisionResult.Value
        let confidence: Double
        switch question {
        case .choice:
            value = .choice(prepared.optionLabels[topIndex])
            confidence = entropyConfidence
        case .score:
            let score = probabilities.enumerated().reduce(0) {
                $0 + Double($1.offset) * $1.element
            }
            value = .score(rounded(score))
            confidence = entropyConfidence
        case .boolean:
            value = .boolean(probabilities[1] >= probabilities[0])
            confidence = max(probabilities[0], probabilities[1])
        }
        return DecisionResult(
            questionName: question.name,
            value: value,
            confidence: rounded(confidence),
            answerConfidence: rounded(answerConfidence),
            probabilities: probabilityMap,
            actProbability: rounded(action[0])
        )
    }

    static func softmax(_ values: [Double]) -> [Double] {
        guard let maximum = values.max() else { return [] }
        let exponentials = values.map { exp($0 - maximum) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    static func rounded(_ value: Double) -> Double {
        (value * 10_000).rounded() / 10_000
    }
}
