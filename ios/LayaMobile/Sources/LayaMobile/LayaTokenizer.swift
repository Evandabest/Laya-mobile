import Foundation
import Tokenizers

public struct LayaTokenizer: Sendable {
    private let backend: any Tokenizer

    public let clsTokenID: Int
    public let sepTokenID: Int
    public let padTokenID: Int
    public let maskTokenID: Int
    public let maskToken = "[MASK]"

    private init(backend: any Tokenizer) throws {
        self.backend = backend
        guard let clsTokenID = backend.convertTokenToId("[CLS]"),
              let sepTokenID = backend.convertTokenToId("[SEP]"),
              let padTokenID = backend.convertTokenToId("[PAD]"),
              let maskTokenID = backend.convertTokenToId("[MASK]")
        else {
            throw LayaError.tokenizer("Tokenizer assets are missing Laya's special tokens")
        }
        self.clsTokenID = clsTokenID
        self.sepTokenID = sepTokenID
        self.padTokenID = padTokenID
        self.maskTokenID = maskTokenID
    }

    public static func load(from folder: URL) async throws -> LayaTokenizer {
        do {
            // This reads tokenizer_config.json and tokenizer.json from the supplied local folder.
            // strict:false permits the generic PreTrainedTokenizerFast class name to select the
            // serialized BPE implementation; no Hub request is made by this initializer.
            let backend = try await AutoTokenizer.from(modelFolder: folder, strict: false)
            return try LayaTokenizer(backend: backend)
        } catch let error as LayaError {
            throw error
        } catch {
            throw LayaError.tokenizer("Unable to load local tokenizer: \(error)")
        }
    }

    public func encode(_ text: String) -> [Int] {
        backend.encode(text: text, addSpecialTokens: false)
    }
}
