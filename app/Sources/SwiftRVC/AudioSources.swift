import AppKit
import AVFoundation
import CoreAudio
import Foundation

enum InputSourceMode: String, CaseIterable, Identifiable {
    case mic = "mic"
    case file = "file"
    case systemApp = "system_app"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .mic: return "Mic"
        case .file: return "File"
        case .systemApp: return "App"
        }
    }
}

struct TappableAppInfo: Identifiable, Hashable {
    static let allSystemID = "__ALL_SYSTEM_AUDIO__"

    let id: String
    let name: String
    let bundleID: String?
    let pid: pid_t
    let isPlaying: Bool

    var isGlobalSystem: Bool { id == Self.allSystemID }

    var displayLabel: String {
        if isGlobalSystem { return name }
        return isPlaying ? "\(name) (playing)" : name
    }
}
	
enum AppAudioDiscovery {
    // used to exclude our own app from global taps so we dont feedback loop
    static func selfProcessObjectID() -> AudioObjectID? {
        if #available(macOS 14.2, *) {
            var myPID = ProcessInfo.processInfo.processIdentifier
            var objID: AudioObjectID = kAudioObjectUnknown
            var size = UInt32(MemoryLayout<AudioObjectID>.size)
            var addr = AudioObjectPropertyAddress(
                mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            
            let st = AudioObjectGetPropertyData(
                AudioObjectID(kAudioObjectSystemObject),
                &addr,
                UInt32(MemoryLayout<pid_t>.size), &myPID,
                &size, &objID
            )
            return (st == noErr && objID != kAudioObjectUnknown) ? objID : nil
        }
        return nil
    }

    private static func allAudioProcessObjects() -> [(obj: AudioObjectID, pid: pid_t, bundleID: String, isPlaying: Bool)] {
        guard #available(macOS 14.2, *) else { return [] }
        
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size
        ) == noErr, size > 0 else {
            return []
        }

        let count = Int(size) / MemoryLayout<AudioObjectID>.size
        var objs = [AudioObjectID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &objs
        ) == noErr else {
            return []
        }
		
        var out: [(AudioObjectID, pid_t, String, Bool)] = []
        for obj in objs {
            var pidAddr = AudioObjectPropertyAddress(
                mSelector: kAudioProcessPropertyPID,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var pid: pid_t = 0
            var psz = UInt32(MemoryLayout<pid_t>.size)
            _ = AudioObjectGetPropertyData(obj, &pidAddr, 0, nil, &psz, &pid)
            
            var bidAddr = AudioObjectPropertyAddress(
                mSelector: kAudioProcessPropertyBundleID,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var cfStr: Unmanaged<CFString>?
            var bsz = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
            let bst = AudioObjectGetPropertyData(obj, &bidAddr, 0, nil, &bsz, &cfStr)
            let bid = (bst == noErr) ? (cfStr?.takeRetainedValue() as String? ?? "") : ""

            var playAddr = AudioObjectPropertyAddress(
                mSelector: kAudioProcessPropertyIsRunningOutput,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )
            var playing: UInt32 = 0
            var plsz = UInt32(MemoryLayout<UInt32>.size)
            _ = AudioObjectGetPropertyData(obj, &playAddr, 0, nil, &plsz, &playing)

            out.append((obj, pid, bid, playing != 0))
        }
        return out
    }

    static func tappableSources() -> [TappableAppInfo] {
        let myPID = ProcessInfo.processInfo.processIdentifier
        let audioProcs = allAudioProcessObjects()

        var playingPIDs = Set<pid_t>()
        var playingBundles = Set<String>()
        for p in audioProcs where p.isPlaying {
            playingPIDs.insert(p.pid)
            if !p.bundleID.isEmpty { playingBundles.insert(p.bundleID) }
        }
        
        var apps: [TappableAppInfo] = []
        var seenIDs = Set<String>()
        var coveredBundles = Set<String>()
        var coveredPIDs = Set<pid_t>()

        for app in NSWorkspace.shared.runningApplications {
            guard app.activationPolicy == .regular else { continue }
            guard app.processIdentifier != myPID else { continue }
            let name = app.localizedName ?? "App \(app.processIdentifier)"
            let bid = app.bundleIdentifier
            let key = bid ?? "pid:\(app.processIdentifier)"
            if seenIDs.contains(key) { continue }
            seenIDs.insert(key)
            coveredPIDs.insert(app.processIdentifier)
            if let bid { coveredBundles.insert(bid) }

            let isPlaying = playingPIDs.contains(app.processIdentifier)
                || (bid != nil && playingBundles.contains { $0 == bid! || $0.hasPrefix(bid! + ".") })

            apps.append(TappableAppInfo(
                id: key,
                name: name,
                bundleID: bid,
                pid: app.processIdentifier,
                isPlaying: isPlaying
            ))
        }
        
        // catch cli/background players like afplay that arent regular gui apps
        for p in audioProcs where p.isPlaying && p.pid != myPID && !coveredPIDs.contains(p.pid) {
            if !p.bundleID.isEmpty && coveredBundles.contains(where: { p.bundleID == $0 || p.bundleID.hasPrefix($0 + ".") }) {
                continue
            }
            let key = p.bundleID.isEmpty ? "pid:\(p.pid)" : p.bundleID
            if seenIDs.contains(key) { continue }
            seenIDs.insert(key)
            let running = NSRunningApplication(processIdentifier: p.pid)
            let label = running?.localizedName ?? (p.bundleID.isEmpty ? "Audio Process (\(p.pid))" : p.bundleID)
            apps.append(TappableAppInfo(
                id: key,
                name: label,
                bundleID: p.bundleID.isEmpty ? nil : p.bundleID,
                pid: p.pid,
                isPlaying: true
            ))
        }

        apps.sort { a, b in
            if a.isPlaying != b.isPlaying { return a.isPlaying }
            return a.name.localizedCaseInsensitiveCompare(b.name) == .orderedAscending
        }

        let globalItem = TappableAppInfo(
            id: TappableAppInfo.allSystemID,
            name: "System Audio",
            bundleID: nil,
            pid: 0,
            isPlaying: false
        )
        return [globalItem] + apps
    }

    // grab main pid plus helper/renderer sub-processes (browsers etc)
    static func resolveProcessObjects(for target: TappableAppInfo) -> [AudioObjectID] {
        let audioProcs = allAudioProcessObjects()
        var matched: [AudioObjectID] = []
        
        for p in audioProcs {
            if p.pid == target.pid {
                matched.append(p.obj)
                continue
            }
            if let bid = target.bundleID, !bid.isEmpty, !p.bundleID.isEmpty {
                if p.bundleID == bid || p.bundleID.hasPrefix(bid + ".") {
                    matched.append(p.obj)
                }
            }
        }
        return matched
    }
}
	
    
		
    
enum AudioFileDecoder {
    static func decodeTo48kMono(url: URL) throws -> [Float] {
        // try native avaudiofile first, fall back to ffmpeg for mp4/ogg/opus/webm
        if let native = try? decodeNativeAVAudioFile(url: url), !native.isEmpty {
            return native
        }
        return try decodeViaFFmpeg(url: url)
    }

    private static func decodeNativeAVAudioFile(url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let inFormat = file.processingFormat
        
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 48000.0,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "Cannot construct 48kHz mono format"])
        }

        guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            throw NSError(domain: "SwiftRVC", code: -2, userInfo: [NSLocalizedDescriptionKey: "Cannot create AVAudioConverter"])
        }

        let inChunkFrames: AVAudioFrameCount = 8192
        let ratio = 48000.0 / max(inFormat.sampleRate, 8000.0)
        let outChunkFrames = AVAudioFrameCount(Double(inChunkFrames) * ratio + 1024)
		
        guard let inBuf = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: inChunkFrames),
              let outBuf = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outChunkFrames) else {
            throw NSError(domain: "SwiftRVC", code: -3, userInfo: [NSLocalizedDescriptionKey: "Buffer allocation failed"])
        }

        var result: [Float] = []
        result.reserveCapacity(Int(Double(file.length) * ratio) + 4096)

        var reachedEOF = false
        while !reachedEOF {
            var convError: NSError?
            let status = converter.convert(to: outBuf, error: &convError) { _, outStatus in
                if reachedEOF {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                do {
                    try file.read(into: inBuf)
                    if inBuf.frameLength == 0 {
                        reachedEOF = true
                        outStatus.pointee = .endOfStream
                        return nil
                    }
                    outStatus.pointee = .haveData
                    return inBuf
                } catch {
                    reachedEOF = true
                    outStatus.pointee = .endOfStream
                    return nil
                }
            }
            if let err = convError { throw err }
            if outBuf.frameLength > 0, let ch = outBuf.floatChannelData?[0] {
                result.append(contentsOf: UnsafeBufferPointer(start: ch, count: Int(outBuf.frameLength)))
            }
            if status == .endOfStream || status == .error { break }
        }
        return result
    }

    private static func decodeViaFFmpeg(url: URL) throws -> [Float] {
        let ffmpegCandidates = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]
        guard let ffmpegPath = ffmpegCandidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            throw NSError(domain: "SwiftRVC", code: -4, userInfo: [NSLocalizedDescriptionKey: "Unsupported format (install ffmpeg for video/ogg)"])
        }
        
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: ffmpegPath)
        proc.arguments = [
            "-v", "error",
            "-i", url.path,
            "-f", "f32le",
            "-ac", "1",
            "-ar", "48000",
            "pipe:1"
        ]
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        try proc.run()
        
        let data = outPipe.fileHandleForReading.readDataToEndOfFile()
        proc.waitUntilExit()

        guard proc.terminationStatus == 0, !data.isEmpty else {
            let errStr = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? "decode failed"
            throw NSError(domain: "SwiftRVC", code: Int(proc.terminationStatus), userInfo: [NSLocalizedDescriptionKey: "ffmpeg failed: \(errStr)"])
        }

        let count = data.count / MemoryLayout<Float>.size
        var samples = [Float](repeating: 0, count: count)
        _ = samples.withUnsafeMutableBytes { data.copyBytes(to: $0) }
        return samples
    }

    static func writeWav48kMono(samples: [Float], to url: URL) throws {
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 48000.0,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "Format error"])
        }
        
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: 48000.0,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        let outFile = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        
        let chunkSize = 32768
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(chunkSize)) else {
            throw NSError(domain: "SwiftRVC", code: -2, userInfo: [NSLocalizedDescriptionKey: "Buffer alloc failed"])
        }
        var offset = 0
        while offset < samples.count {
            let n = min(chunkSize, samples.count - offset)
            buf.frameLength = AVAudioFrameCount(n)
            if let dst = buf.floatChannelData?[0] {
                for i in 0..<n {
                    dst[i] = max(-1.0, min(1.0, samples[offset + i]))
                }
            }
            try outFile.write(from: buf)
            offset += n
        }
    }
}
