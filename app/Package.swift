// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "SwiftRVC",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        .executable(
            name: "SwiftRVC",
            targets: ["SwiftRVC"]
        )
    ],
    targets: [
        .executableTarget(
            name: "SwiftRVC",
            path: "Sources/SwiftRVC",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("SwiftUI"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Accelerate")
            ]
        )
    ]
)
