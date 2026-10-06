import LayaMobile
import SwiftUI

private enum DemoQuestionSet: String, CaseIterable, Identifiable {
    case route = "Choice · Route"
    case urgency = "Score · Urgency"
    case refund = "Boolean · Refund"
    case verified = "Boolean · Custom labels"
    case all = "Multiple · All primary types"

    var id: Self { self }

    var questions: [LayaQuestion] {
        switch self {
        case .route: [Self.routeQuestion]
        case .urgency: [Self.urgencyQuestion]
        case .refund: [Self.refundQuestion]
        case .verified: [Self.verifiedQuestion]
        case .all: [Self.routeQuestion, Self.urgencyQuestion, Self.refundQuestion]
        }
    }

    private static let routeQuestion = LayaQuestion.choice(
        .init(
            name: "department",
            instructions: "Which department should handle this request?",
            options: [
                .init("billing", criterion: "invoices, payments, refunds, or duplicate charges"),
                .init("technical", criterion: "bugs, outages, login failures, or system errors"),
                .init("sales", criterion: "pricing, plans, demos, or new contracts"),
                .init("other", criterion: "requests that do not match another department"),
            ]
        )
    )

    private static let urgencyQuestion = LayaQuestion.score(
        .init(
            name: "urgency",
            instructions: "How urgent is this request?",
            levels: [
                "not urgent; no deadline or impact",
                "soon; should be handled promptly",
                "critical; deadline, outage, or blocking issue",
            ]
        )
    )

    private static let refundQuestion = LayaQuestion.boolean(
        .init(
            name: "refund",
            instructions: "Does the customer ask for money back?",
            falseCriterion: "the customer does not request a refund",
            trueCriterion: "the customer explicitly requests a refund or reimbursement"
        )
    )

    private static let verifiedQuestion = LayaQuestion.boolean(
        .init(
            name: "identity_verified",
            instructions: "Has the customer completed identity verification?",
            falseCriterion: "verification is missing, incomplete, or failed",
            trueCriterion: "verification is explicitly confirmed as complete",
            falseLabel: "needs_review",
            trueLabel: "verified"
        )
    )
}

private enum SampleScenario: String, CaseIterable, Identifiable {
    case custom = "Custom input"
    case refundText = "Refund request"
    case technicalObject = "Structured support ticket"
    case conversation = "Conversation history"
    case criticalScore = "Critical incident"
    case customBoolean = "Custom boolean labels"
    case unicode = "Unicode text"
    case longConversation = "Long conversation"
    case empty = "Empty input"

    var id: Self { self }

    var summary: String {
        switch self {
        case .custom: "Type any text · choose any output mode"
        case .refundText: "Plain text · three questions in one call"
        case .technicalObject: "Ordered JSON object · choice output"
        case .conversation: "Chronological turn list · all output types"
        case .criticalScore: "Plain text · expected rubric score"
        case .customBoolean: "Custom true/false labels and criteria"
        case .unicode: "Emoji, accents, and mixed scripts"
        case .longConversation: "Newest-turn-preserving truncation"
        case .empty: "Minimal valid text input"
        }
    }

    var questionSet: DemoQuestionSet {
        switch self {
        case .custom, .refundText, .conversation, .longConversation, .empty: .all
        case .technicalObject, .unicode: .route
        case .criticalScore: .urgency
        case .customBoolean: .verified
        }
    }

    var state: LayaState {
        switch self {
        case .custom:
            .text("")
        case .refundText:
            .text("I was charged twice. Please refund the duplicate payment today.")
        case .technicalObject:
            .object([
                .init("ticket_id", .string("IOS-1042")),
                .init("subject", .string("Cannot sign in after update")),
                .init(
                    "message",
                    .string("The app shows an authentication error every time I sign in.")
                ),
                .init("plan", .string("pro")),
                .init("attempts", .integer(4)),
            ])
        case .conversation:
            .conversation([
                Self.turn("user", "I need help with a duplicate card charge."),
                Self.turn("assistant", "I can help. Are you requesting a refund?"),
                Self.turn("user", "Yes. Please refund it before my rent is due tomorrow."),
            ])
        case .criticalScore:
            .text(
                "Production is down for every customer. Checkout is failing and our launch "
                    + "starts in twenty minutes. We need immediate help."
            )
        case .customBoolean:
            .object([
                .init("customer", .string("Ada")),
                .init("verification_status", .string("completed")),
                .init("verified_at", .string("2026-10-03T22:40:00Z")),
            ])
        case .unicode:
            .text("Facturé deux fois 😟 — 请帮我退款. Order café-東京-42.")
        case .longConversation:
            .conversation(
                (0..<28).flatMap { index in
                    [
                        Self.turn(
                            "user",
                            "Earlier message \(index): background details about my account and order."
                        ),
                        Self.turn("assistant", "Acknowledged earlier message \(index)."),
                    ]
                } + [
                    Self.turn(
                        "user",
                        "LATEST REQUEST: production is blocked and I need the duplicate charge "
                            + "refunded immediately."
                    ),
                ]
            )
        case .empty:
            .text("")
        }
    }

    private static func turn(_ role: String, _ content: String) -> LayaJSONValue {
        .object([
            .init("role", .string(role)),
            .init("content", .string(content)),
        ])
    }
}

@MainActor
private final class DemoModel: ObservableObject {
    @Published var scenario: SampleScenario = .refundText
    @Published var questionSet: DemoQuestionSet = .all
    @Published var state: LayaState = SampleScenario.refundText.state
    @Published var status = "Loading local model…"
    @Published var detail = ""
    @Published var prediction: LayaPrediction?
    @Published var lastQuestions: [LayaQuestion] = []
    @Published var isReady = false
    @Published var isRunning = false

    private var laya: LayaModel?

    var stateKind: String {
        switch state {
        case .text: "Text"
        case .object: "JSON object"
        case .conversation: "Conversation list"
        }
    }

    var statePreview: String {
        (try? state.serialized()) ?? "Unable to serialize state"
    }

    var editableText: String {
        get {
            guard case .text(let value) = state else { return statePreview }
            return value
        }
        set {
            guard case .text(let current) = state, current != newValue else { return }
            scenario = .custom
            state = .text(newValue)
            clearOutput()
        }
    }

    var isTextState: Bool {
        if case .text = state { return true }
        return false
    }

    func apply(_ scenario: SampleScenario) {
        self.scenario = scenario
        state = scenario.state
        questionSet = scenario.questionSet
        clearOutput()
    }

    func selectQuestions(_ set: DemoQuestionSet) {
        questionSet = set
        clearOutput()
    }

    func load() async {
        guard let resources = Bundle.main.resourceURL else {
            status = "Model bundle unavailable"
            return
        }
        let bundle = resources.appending(path: "LayaModel/Generated", directoryHint: .isDirectory)
        guard Self.hasGeneratedModelBundle(at: bundle) else {
            status = "Model resources are not installed"
            detail = "From the repository root, run: "
                + ".venv-coreml/bin/python -m conversion.export_coreml "
                + "--output ios/ExampleApp/Resources/LayaModel/Generated, then rebuild the app."
            return
        }
        do {
            laya = try await LayaModel.load(from: bundle)
            status = "Ready — fully offline"
            detail = LayaModel.backend
            isReady = true
        } catch {
            status = "Model is not installed"
            detail = error.localizedDescription
        }
    }

    func predict() async {
        guard let laya else { return }
        isRunning = true
        prediction = nil
        status = "Running locally…"
        let count = questionSet.questions.count
        detail = count == 1
            ? "Processing 1 question on Core ML."
            : "Processing \(count) questions sequentially on Core ML."
        defer { isRunning = false }
        do {
            let questions = questionSet.questions
            let state = state
            let result = try await Task.detached(priority: .userInitiated) {
                try laya.predict(state: state, questions: questions)
            }.value
            lastQuestions = questions
            prediction = result
            status = result.results.count == 1
                ? "Completed 1 decision"
                : "Completed \(result.results.count) decisions"
            detail = "All values below are from the local Core ML result."
        } catch {
            status = "Inference failed"
            detail = error.localizedDescription
        }
    }

    private func clearOutput() {
        prediction = nil
        lastQuestions = []
        if isReady {
            status = "Ready — fully offline"
            detail = LayaModel.backend
        }
    }

    private static func hasGeneratedModelBundle(at bundle: URL) -> Bool {
        let files = FileManager.default
        let requiredFiles = [
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "rl_agent_config.json",
        ]
        guard requiredFiles.allSatisfy({
            files.fileExists(atPath: bundle.appending(path: $0).path)
        }) else {
            return false
        }
        return ["laya.mlpackage", "laya.mlmodelc"].contains {
            files.fileExists(atPath: bundle.appending(path: $0).path)
        }
    }
}

struct ContentView: View {
    @StateObject private var model = DemoModel()
    @FocusState private var inputFocused: Bool

    var body: some View {
        NavigationStack {
            Form {
                runtimeSection
                examplesSection
                inputSection
                questionsSection
                runSection
                outputSection
            }
            .navigationTitle("Laya Mobile")
            .task { await model.load() }
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { inputFocused = false }
                }
            }
        }
    }

    private var runtimeSection: some View {
        Section {
            Label("On-device · no network", systemImage: "iphone.and.arrow.forward")
                .foregroundStyle(.green)
            LabeledContent("Runtime", value: model.isReady ? LayaModel.backend : "Loading…")
        }
    }

    private var examplesSection: some View {
        Section("Examples") {
            Picker("Test case", selection: Binding(
                get: { model.scenario },
                set: { model.apply($0) }
            )) {
                ForEach(SampleScenario.allCases) { scenario in
                    Text(scenario.rawValue).tag(scenario)
                }
            }
            Text(model.scenario.summary)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var inputSection: some View {
        Section {
            LabeledContent("State type", value: model.stateKind)
            if model.isTextState {
                ZStack(alignment: .topLeading) {
                    if model.editableText.isEmpty {
                        Text("Type or paste any text for Laya to evaluate…")
                            .foregroundStyle(.tertiary)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 8)
                            .allowsHitTesting(false)
                    }
                    TextEditor(text: Binding(
                        get: { model.editableText },
                        set: { model.editableText = $0 }
                    ))
                    .focused($inputFocused)
                    .font(.body)
                    .scrollContentBackground(.hidden)
                }
                .frame(minHeight: 130)

                Button("Clear input", systemImage: "xmark.circle") {
                    model.editableText = ""
                    inputFocused = true
                }
                .disabled(model.editableText.isEmpty)
            } else {
                Text(model.statePreview)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
            }
        } header: {
            Text("Input sent to Laya")
        } footer: {
            if !model.isTextState {
                Text("Structured examples show the exact upstream-compatible JSON tokenized by the model.")
            }
        }
    }

    private var questionsSection: some View {
        Section("Output mode") {
            Picker("Questions", selection: Binding(
                get: { model.questionSet },
                set: { model.selectQuestions($0) }
            )) {
                ForEach(DemoQuestionSet.allCases) { set in
                    Text(set.rawValue).tag(set)
                }
            }

            ForEach(Array(model.questionSet.questions.enumerated()), id: \.element.name) {
                _, question in
                QuestionSummary(question: question)
            }
        }
    }

    private var runSection: some View {
        Section {
            Button {
                Task { await model.predict() }
            } label: {
                HStack {
                    Spacer()
                    if model.isRunning { ProgressView() }
                    Text(model.isRunning ? "Running locally…" : "Run locally")
                    Spacer()
                }
            }
            .disabled(!model.isReady || model.isRunning)

            if model.isRunning {
                HStack(spacing: 12) {
                    ProgressView()
                        .controlSize(.large)
                    VStack(alignment: .leading, spacing: 3) {
                        Text("Inference in progress")
                            .fontWeight(.semibold)
                        Text("The simulator can take several seconds per question.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(.vertical, 6)
            }

            Text(model.status)
                .font(.headline)
            if !model.detail.isEmpty {
                Text(model.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder
    private var outputSection: some View {
        if let prediction = model.prediction {
            Section("Output") {
                ForEach(Array(prediction.results.enumerated()), id: \.element.questionName) {
                    index, result in
                    let question = model.lastQuestions[index]
                    ResultCard(result: result, question: question)
                }
            }

            Section("Inference details") {
                LabeledContent("Backend", value: prediction.backend)
                LabeledContent(
                    "Total request latency",
                    value: prediction.latency.formattedMilliseconds
                )
                LabeledContent("Input tokens", value: String(prediction.usage.inputTokens))
                LabeledContent("State tokens", value: String(prediction.usage.stateTokens))
                LabeledContent(
                    "Dropped state tokens",
                    value: String(prediction.usage.stateTokensDropped)
                )
                LabeledContent("Truncated", value: prediction.usage.truncated ? "Yes" : "No")
            }
        }
    }
}

private struct QuestionSummary: View {
    let question: LayaQuestion

    var body: some View {
        DisclosureGroup("\(question.typeName) · \(question.name)") {
            VStack(alignment: .leading, spacing: 8) {
                Text(question.instructions)
                    .frame(maxWidth: .infinity, alignment: .leading)
                ForEach(question.optionDescriptions, id: \.label) { option in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(option.label).fontWeight(.semibold)
                        if let detail = option.detail {
                            Text(detail).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding(.top, 6)
        }
    }
}

private struct ResultCard: View {
    let result: DecisionResult
    let question: LayaQuestion

    private var selectedLabel: String {
        result.displayValue(for: question)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(result.questionName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(selectedLabel)
                        .font(.title3.weight(.semibold))
                }
                Spacer()
                Text(question.typeName)
                    .font(.caption.weight(.medium))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.blue.opacity(0.12), in: Capsule())
            }

            VStack(spacing: 10) {
                ForEach(question.probabilityLabels, id: \.self) { label in
                    ProbabilityRow(
                        label: question.displayLabel(forProbabilityKey: label),
                        probability: result.probabilities[label] ?? 0,
                        isHighest: label == result.highestProbabilityLabel
                    )
                }
            }

            Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 5) {
                GridRow {
                    Text("Question latency").foregroundStyle(.secondary)
                    Text(result.latency.formattedMilliseconds)
                        .monospacedDigit()
                }
                metricRow("Confidence", result.confidence)
                metricRow("Answer confidence", result.answerConfidence)
                metricRow("Act probability", result.actProbability)
            }
            .font(.caption)
        }
        .padding(.vertical, 6)
    }

    private func metricRow(_ name: String, _ value: Double) -> some View {
        GridRow {
            Text(name).foregroundStyle(.secondary)
            Text(value.formatted(.number.precision(.fractionLength(4))))
                .monospacedDigit()
        }
    }
}

private struct ProbabilityRow: View {
    let label: String
    let probability: Double
    let isHighest: Bool

    var body: some View {
        VStack(spacing: 4) {
            HStack {
                Text(label)
                    .fontWeight(isHighest ? .semibold : .regular)
                Spacer()
                Text(probability, format: .percent.precision(.fractionLength(2)))
                    .monospacedDigit()
            }
            ProgressView(value: probability)
                .tint(isHighest ? .blue : .secondary)
        }
    }
}

private extension LayaQuestion {
    var typeName: String {
        switch self {
        case .choice: "choice"
        case .score: "score"
        case .boolean: "boolean / noul"
        }
    }

    var instructions: String {
        switch self {
        case .choice(let value): value.instructions
        case .score(let value): value.instructions
        case .boolean(let value): value.instructions
        }
    }

    var optionDescriptions: [(label: String, detail: String?)] {
        switch self {
        case .choice(let value):
            value.options.map { ($0.label, $0.criterion) }
        case .score(let value):
            value.levels.enumerated().map { ("level \($0.offset)", $0.element) }
        case .boolean(let value):
            [
                (value.falseLabel, value.falseCriterion ?? "statement does not hold"),
                (value.trueLabel, value.trueCriterion ?? "statement holds"),
            ]
        }
    }

    var probabilityLabels: [String] {
        switch self {
        case .choice(let value): value.options.map(\.label)
        case .score(let value): value.levels.indices.map(String.init)
        case .boolean(let value): [value.falseLabel, value.trueLabel]
        }
    }

    func displayLabel(forProbabilityKey key: String) -> String {
        guard case .score(let value) = self,
              let index = Int(key),
              value.levels.indices.contains(index)
        else {
            return key
        }
        return "\(key) · \(value.levels[index])"
    }
}

private extension DecisionResult {
    var highestProbabilityLabel: String? {
        probabilities.max { $0.value < $1.value }?.key
    }

    func displayValue(for question: LayaQuestion) -> String {
        switch value {
        case .choice(let value):
            return value
        case .score(let value):
            let maximum = max(0, question.probabilityLabels.count - 1)
            return "Expected score \(value.formatted(.number.precision(.fractionLength(4)))) / \(maximum)"
        case .boolean(let value):
            guard case .boolean(let definition) = question else {
                return value ? "true" : "false"
            }
            return value ? definition.trueLabel : definition.falseLabel
        }
    }
}

private extension Duration {
    var formattedMilliseconds: String {
        let parts = components
        let milliseconds = Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
        return milliseconds.formatted(.number.precision(.fractionLength(1))) + " ms"
    }
}
