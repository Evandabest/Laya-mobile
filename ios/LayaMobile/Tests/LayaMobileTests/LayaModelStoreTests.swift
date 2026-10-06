import XCTest
@testable import LayaMobile

private actor LoadCounter {
    private(set) var count = 0

    func load() async throws -> Int {
        count += 1
        try await Task.sleep(for: .milliseconds(20))
        return 42
    }
}

final class LayaModelStoreTests: XCTestCase {
    func testCacheLoadsLazilyAndCoalescesConcurrentRequests() async throws {
        let counter = LoadCounter()
        let cache = IdleResourceCache<Int>(idleTimeout: .seconds(10)) {
            try await counter.load()
        }

        let initiallyLoaded = await cache.containsValue()
        XCTAssertFalse(initiallyLoaded)
        async let first = cache.acquire()
        async let second = cache.acquire()

        let values = try await [first, second]
        XCTAssertEqual(values, [42, 42])
        let loadCount = await counter.count
        let isLoaded = await cache.containsValue()
        XCTAssertEqual(loadCount, 1)
        XCTAssertTrue(isLoaded)
    }

    func testCacheEvictsAfterIdleTimeout() async throws {
        let cache = IdleResourceCache<Int>(idleTimeout: .milliseconds(20)) { 42 }

        _ = try await cache.acquire()
        await cache.release()
        try await Task.sleep(for: .milliseconds(60))

        let isLoaded = await cache.containsValue()
        XCTAssertFalse(isLoaded)
    }

    func testExplicitEvictionForcesNextAcquireToReload() async throws {
        let counter = LoadCounter()
        let cache = IdleResourceCache<Int>(idleTimeout: .seconds(10)) {
            try await counter.load()
        }

        _ = try await cache.acquire()
        await cache.evict()
        let isLoadedAfterEviction = await cache.containsValue()
        XCTAssertFalse(isLoadedAfterEviction)
        _ = try await cache.acquire()

        let loadCount = await counter.count
        XCTAssertEqual(loadCount, 2)
    }
}
