import LayaMobile
import SwiftUI

private enum SampleQuestion: String, CaseIterable, Identifiable {
    case choice = "Route"
    case score = "Urgency"
    case boolean = "Refund"

    var id: Self { self }

    var question: LayaQuestion {
        switch self {
        case .choice:
            .choice(
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
            )
        case .score:
            .score(
                .init(
                    name: "urgency",
                    instructions: "How urgent is this request?",
                    levels: ["not urgent", "soon", "critical deadline or blocking issue"]
                )
            )
        case .boolean:
            .boolean(
                .init(name: "refund", instructions: "Does the customer ask for money back?")
            )
        }
    }
}

@MainActor
private final class DemoModel: ObservableObject {
    @Published var text = "I was charged twice. Please refund the duplicate payment."
    @Published var sample: SampleQuestion = .choice
    @Published var status = "Loading local model…"
    @Published var detail = ""
    @Published var isReady = false
    @Published var isRunning = false

    private var laya: LayaModel?

    func load() async {
        guard let resources = Bundle.main.resourceURL else {
            status = "Model bundle unavailable"
            return
        }
        let bundle = resources.appending(path: "LayaModel/Generated", directoryHint: .isDirectory)
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
        defer { isRunning = false }
        do {
            let prediction = try laya.predict(text: text, questions: [sample.question])
            guard let result = prediction.results.first else { return }
            status = result.value.displayText
            let milliseconds = prediction.latency.milliseconds
            detail = String(
                format: "confidence %.4f  •  %.1f ms  •  %@%@",
                result.confidence,
                milliseconds,
                prediction.backend,
                prediction.usage.truncated ? "  •  truncated" : ""
            )
        } catch {
            status = "Inference failed"
            detail = error.localizedDescription
        }
    }
}

private extension DecisionResult.Value {
    var displayText: String {
        switch self {
        case .choice(let value): value
        case .score(let value): String(format: "%.4f", value)
        case .boolean(let value): value ? "yes" : "no"
        }
    }
}

private extension Duration {
    var milliseconds: Double {
        let parts = components
        return Double(parts.seconds) * 1_000 + Double(parts.attoseconds) / 1e15
    }
}

struct ContentView: View {
    @StateObject private var model = DemoModel()

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Label("On-device · no network", systemImage: "iphone.and.arrow.forward")
                        .foregroundStyle(.green)
                    TextEditor(text: $model.text)
                        .frame(minHeight: 150)
                } header: {
                    Text("Input")
                }

                Section("Decision") {
                    Picker("Question", selection: $model.sample) {
                        ForEach(SampleQuestion.allCases) { sample in
                            Text(sample.rawValue).tag(sample)
                        }
                    }
                    .pickerStyle(.segmented)

                    Button {
                        Task { await model.predict() }
                    } label: {
                        HStack {
                            if model.isRunning { ProgressView() }
                            Text(model.isRunning ? "Running…" : "Run locally")
                        }
                        .frame(maxWidth: .infinity)
                    }
                    .disabled(!model.isReady || model.isRunning || model.text.isEmpty)
                }

                Section("Result") {
                    Text(model.status).font(.headline)
                    if !model.detail.isEmpty {
                        Text(model.detail).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("Laya Mobile")
            .task { await model.load() }
        }
    }
}
