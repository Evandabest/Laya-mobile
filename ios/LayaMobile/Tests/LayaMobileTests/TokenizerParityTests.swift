import Foundation
import Testing
@testable import LayaMobile

private func repositoryRoot() -> URL {
    URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
}

private func tokenizerURL() -> URL? {
    if let path = ProcessInfo.processInfo.environment["LAYA_TOKENIZER_DIR"] {
        return URL(fileURLWithPath: path)
    }
    let local = repositoryRoot().appending(path: "models/laya/tokenizer")
    return FileManager.default.fileExists(atPath: local.path) ? local : nil
}

@Test func tokenizerMatchesReferenceUnicodeAndWhitespace() async throws {
    guard let folder = tokenizerURL() else { return }
    let tokenizer = try await LayaTokenizer.load(from: folder)
    let cases: [(String, [Int])] = [
        ("I was charged twice.", [42, 369, 6636, 7019, 15]),
        ("你好，世界", [24553, 34439, 6238, 42848, 45261]),
        ("emoji 👩🏽‍💻🚀", [43208, 8020, 22692, 228, 104, 14931, 226, 123, 325, 224, 14931, 229, 121, 14931, 237, 211]),
        ("  tabs\tand\nlines  ", [50276, 33754, 186, 395, 187, 8737, 50276]),
        ("café", [68, 2320, 860]),
        ("cafe\u{0301}", [68, 2320, 860]),
        ("مرحبا بالعالم", [5843, 6900, 21931, 13621, 3142, 15677, 7427, 13793, 7427, 5843]),
        ("हैलो दुनिया", [30598, 33358, 29903, 25159, 6280, 101, 38619, 23068, 20489, 29950, 12001]),
        ("日本語のテスト", [49868, 19119, 241, 3917, 28656, 38206]),
        ("Привет, мир!", [21715, 1697, 16423, 7508, 13, 8277, 20650, 2]),
        ("punctuation?!…—“quotes”", [81, 10593, 2368, 22418, 2866, 1128, 1628, 371, 4787, 668]),
        ("[MASK] literal", [50284, 22436]),
    ]
    for (text, expected) in cases {
        #expect(tokenizer.encode(text) == expected, "Tokenizer mismatch for \(text.debugDescription)")
    }
}

@Test func preprocessingMatchesReferenceFixtureTensors() async throws {
    guard let folder = tokenizerURL() else { return }
    let fixtureURL = repositoryRoot().appending(
        path: "parity_tests/fixtures/laya-ios-english-v1.json"
    )
    let object = try JSONSerialization.jsonObject(with: Data(contentsOf: fixtureURL))
    let bundle = try #require(object as? [String: Any])
    let cases = try #require(bundle["cases"] as? [[String: Any]])
    let fixture = try #require(cases.first { $0["name"] as? String == "empty_state" })
    let rows = try #require(fixture["rows"] as? [[String: Any]])
    let tokenizer = try await LayaTokenizer.load(from: folder)
    let preprocessor = LayaPreprocessor(tokenizer: tokenizer)
    let questions: [String: LayaQuestion] = [
        "department": .choice(
            .init(
                name: "department",
                instructions: "Which department should handle this email?",
                options: [
                    .init("billing", criterion: "invoices, payments, refunds"),
                    .init("technical", criterion: "bugs, outages, system errors"),
                    .init("sales", criterion: "pricing, new contracts"),
                    .init("other", criterion: "everything else"),
                ]
            )
        ),
        "urgency": .score(
            .init(
                name: "urgency",
                instructions: "How urgent is this request?",
                levels: ["not urgent", "soon", "critical deadline or blocking issue"]
            )
        ),
        "refund": .boolean(
            .init(name: "refund", instructions: "Does the customer ask for money back?")
        ),
    ]

    for row in rows {
        let questionID = try #require(row["question_id"] as? String)
        let question = try #require(questions[questionID])
        let actual = try preprocessor.prepare(text: "", question: question)
        #expect(actual.inputIDs == (try #require(row["input_ids"] as? [Int])).map(Int32.init))
        #expect(
            actual.attentionMask
                == (try #require(row["attention_mask"] as? [Bool])).map { $0 ? 1 : 0 }
        )
        #expect(
            actual.markerPositions == (try #require(row["marker_pos"] as? [Int])).map(Int32.init)
        )
        #expect(
            actual.markerMask == (try #require(row["marker_mask"] as? [Bool])).map { $0 ? 1 : 0 }
        )
        #expect(actual.questionType == Int32(try #require(row["qtype"] as? Int)))
    }
}
