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
    targets: [
        .target(name: "LayaMobile"),
        .testTarget(name: "LayaMobileTests", dependencies: ["LayaMobile"]),
    ]
)
