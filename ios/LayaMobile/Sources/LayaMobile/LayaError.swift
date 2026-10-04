import Foundation

public enum LayaError: Error, Equatable, Sendable {
    case invalidQuestion(String)
    case invalidModelBundle(String)
    case unsupportedState(String)
    case tokenizer(String)
    case model(String)
}

extension LayaError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidQuestion(let message),
             .invalidModelBundle(let message),
             .unsupportedState(let message),
             .tokenizer(let message),
             .model(let message):
            message
        }
    }
}
