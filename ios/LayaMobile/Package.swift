// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "LayaMobile",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "LayaMobile", targets: ["LayaMobile"]),
    ],
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "LayaMobile",
            dependencies: [
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .testTarget(name: "LayaMobileTests", dependencies: ["LayaMobile"]),
    ]
)
