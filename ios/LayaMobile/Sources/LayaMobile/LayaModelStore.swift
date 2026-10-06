import Foundation

actor IdleResourceCache<Value: Sendable> {
    typealias Loader = @Sendable () async throws -> Value

    private let idleTimeout: Duration
    private let loader: Loader
    private var cachedValue: Value?
    private var loadingTask: Task<Value, Error>?
    private var loadingGeneration: UInt = 0
    private var generation: UInt = 0
    private var evictionTask: Task<Void, Never>?

    init(idleTimeout: Duration, loader: @escaping Loader) {
        self.idleTimeout = idleTimeout
        self.loader = loader
    }

    func acquire() async throws -> Value {
        evictionTask?.cancel()
        evictionTask = nil

        if let cachedValue {
            return cachedValue
        }

        if let loadingTask {
            return try await finishLoading(loadingTask, generation: loadingGeneration)
        }

        let currentGeneration = generation
        let task = Task { try await loader() }
        loadingTask = task
        loadingGeneration = currentGeneration
        return try await finishLoading(task, generation: currentGeneration)
    }

    func release() {
        evictionTask?.cancel()
        let expectedGeneration = generation
        evictionTask = Task { [weak self, idleTimeout] in
            do {
                try await Task.sleep(for: idleTimeout)
            } catch {
                return
            }
            await self?.evictIfUnchanged(expectedGeneration)
        }
    }

    func evict() {
        generation &+= 1
        evictionTask?.cancel()
        evictionTask = nil
        loadingTask?.cancel()
        loadingTask = nil
        cachedValue = nil
    }

    func containsValue() -> Bool {
        cachedValue != nil
    }

    private func finishLoading(
        _ task: Task<Value, Error>,
        generation expectedGeneration: UInt
    ) async throws -> Value {
        do {
            let loaded = try await task.value
            guard generation == expectedGeneration else {
                throw CancellationError()
            }
            cachedValue = loaded
            loadingTask = nil
            return loaded
        } catch {
            if generation == expectedGeneration {
                loadingTask = nil
            }
            throw error
        }
    }

    private func evictIfUnchanged(_ expectedGeneration: UInt) {
        guard generation == expectedGeneration else { return }
        evict()
    }
}

/// Lazily loads one Core ML model, serializes predictions, and releases it after inactivity.
public actor LayaModelStore {
    public static let defaultIdleTimeout: Duration = .seconds(60)

    private let cache: IdleResourceCache<LayaModel>

    public init(
        bundleURL: URL,
        idleTimeout: Duration = LayaModelStore.defaultIdleTimeout
    ) {
        cache = IdleResourceCache(idleTimeout: idleTimeout) {
            try await LayaModel.load(from: bundleURL)
        }
    }

    public func predict(text: String, questions: [LayaQuestion]) async throws -> LayaPrediction {
        try await predict(state: .text(text), questions: questions)
    }

    public func predict(
        state: LayaState,
        questions: [LayaQuestion]
    ) async throws -> LayaPrediction {
        let model = try await cache.acquire()
        do {
            let prediction = try model.predict(state: state, questions: questions)
            await cache.release()
            return prediction
        } catch {
            await cache.release()
            throw error
        }
    }

    /// Drops the store's model reference. Core ML decides when underlying allocations are reclaimed.
    public func unload() async {
        await cache.evict()
    }

    public func isLoaded() async -> Bool {
        await cache.containsValue()
    }
}
