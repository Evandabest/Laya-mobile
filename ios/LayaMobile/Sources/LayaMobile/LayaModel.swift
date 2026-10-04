import CoreML
import Foundation

public final class LayaModel: @unchecked Sendable {
    public static let backend = "Core ML FP16"

    private let model: MLModel
    private let preprocessor: LayaPreprocessor
    private let calibration: LayaCalibration

    private init(model: MLModel, tokenizer: LayaTokenizer, calibration: LayaCalibration) {
        self.model = model
        self.preprocessor = LayaPreprocessor(tokenizer: tokenizer)
        self.calibration = calibration
    }

    public static func load(from bundleURL: URL) async throws -> LayaModel {
        let tokenizer = try await LayaTokenizer.load(
            from: bundleURL.appending(path: "tokenizer", directoryHint: .isDirectory)
        )
        let calibration = try loadCalibration(
            from: bundleURL.appending(path: "rl_agent_config.json")
        )
        let modelURL = try locateModel(in: bundleURL)
        let compiledURL: URL
        if modelURL.pathExtension == "mlmodelc" {
            compiledURL = modelURL
        } else {
            do {
                compiledURL = try await MLModel.compileModel(at: modelURL)
            } catch {
                throw LayaError.invalidModelBundle("Unable to compile Core ML model: \(error)")
            }
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        do {
            return try LayaModel(
                model: MLModel(contentsOf: compiledURL, configuration: configuration),
                tokenizer: tokenizer,
                calibration: calibration
            )
        } catch {
            throw LayaError.invalidModelBundle("Unable to load Core ML model: \(error)")
        }
    }

    public func predict(text: String, questions: [LayaQuestion]) throws -> LayaPrediction {
        try predict(state: .text(text), questions: questions)
    }

    public func predict(state: LayaState, questions: [LayaQuestion]) throws -> LayaPrediction {
        guard !questions.isEmpty else {
            throw LayaError.invalidQuestion("At least one question is required")
        }
        let clock = ContinuousClock()
        let started = clock.now
        var results: [DecisionResult] = []
        var inputTokens = 0
        var stateTokens = 0
        var dropped = 0
        for question in questions {
            let prepared = try preprocessor.prepare(state: state, question: question)
            let outputs = try prediction(prepared)
            results.append(
                try LayaDecoder.decode(
                    question: question,
                    prepared: prepared,
                    logits: outputs.logits,
                    actionLogits: outputs.actionLogits,
                    calibration: calibration
                )
            )
            inputTokens += prepared.attentionMask.reduce(0) { $0 + Int($1) }
            stateTokens = max(stateTokens, prepared.stateTokens)
            dropped = max(dropped, prepared.stateTokensDropped)
        }
        return LayaPrediction(
            results: results,
            usage: .init(
                inputTokens: inputTokens,
                stateTokens: stateTokens,
                stateTokensDropped: dropped
            ),
            latency: started.duration(to: clock.now),
            backend: Self.backend
        )
    }

    func prediction(_ prepared: PreparedQuestion) throws -> (
        logits: [Double], actionLogits: [Double]
    ) {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": MLFeatureValue(multiArray: try multiArray(prepared.inputIDs, width: 512)),
                "attention_mask": MLFeatureValue(
                    multiArray: try multiArray(prepared.attentionMask, width: 512)
                ),
                "marker_pos": MLFeatureValue(
                    multiArray: try multiArray(prepared.markerPositions, width: 48)
                ),
                "marker_mask": MLFeatureValue(
                    multiArray: try multiArray(prepared.markerMask, width: 48)
                ),
                "qtype": MLFeatureValue(
                    multiArray: try multiArray([prepared.questionType], width: 1)
                ),
            ])
            let output = try model.prediction(from: provider)
            guard let logits = output.featureValue(for: "logits")?.multiArrayValue,
                  let actionLogits = output.featureValue(for: "action_logits")?.multiArrayValue
            else {
                throw LayaError.model("Core ML did not return logits and action_logits")
            }
            return (values(logits), values(actionLogits))
        } catch let error as LayaError {
            throw error
        } catch {
            throw LayaError.model("Core ML prediction failed: \(error)")
        }
    }

    private func multiArray(_ values: [Int32], width: Int) throws -> MLMultiArray {
        let shape: [NSNumber] = width == 1 ? [1] : [1, NSNumber(value: width)]
        let array = try MLMultiArray(shape: shape, dataType: .int32)
        for (index, value) in values.enumerated() {
            array[index] = NSNumber(value: value)
        }
        return array
    }

    private func values(_ array: MLMultiArray) -> [Double] {
        (0..<array.count).map { array[$0].doubleValue }
    }

    private static func locateModel(in bundleURL: URL) throws -> URL {
        for name in ["laya.mlmodelc", "laya.mlpackage"] {
            let candidate = bundleURL.appending(path: name)
            if FileManager.default.fileExists(atPath: candidate.path) {
                return candidate
            }
        }
        throw LayaError.invalidModelBundle("Expected laya.mlmodelc or laya.mlpackage")
    }

    private static func loadCalibration(from url: URL) throws -> LayaCalibration {
        struct Configuration: Decodable {
            let temperature: [Double]
            let temperatureByOptions: [String: Double]

            enum CodingKeys: String, CodingKey {
                case temperature
                case temperatureByOptions = "temperature_by_options"
            }
        }
        do {
            let config = try JSONDecoder().decode(Configuration.self, from: Data(contentsOf: url))
            return LayaCalibration(
                temperatures: config.temperature,
                temperaturesByOptions: config.temperatureByOptions
            )
        } catch {
            throw LayaError.invalidModelBundle("Unable to read calibration configuration: \(error)")
        }
    }
}
