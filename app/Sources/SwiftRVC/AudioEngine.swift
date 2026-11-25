import Accelerate
import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation

// handles mic, file playback, and macos process tap -> unix socket -> output + monitor
final class CoreAudioBridgeEngine: @unchecked Sendable {
    static let targetSampleRate: Double = 48000.0

    private let inRing = FloatRingBuffer(capacity: 48000 * 2)
    private let outRing = FloatRingBuffer(capacity: 48000 * 2)
    private let monRing = FloatRingBuffer(capacity: 48000 * 2)

    private var inputUnit: AudioUnit?
    private var outputUnit: AudioUnit?
    private var monitorUnit: AudioUnit?
    
    // process tap (macos 14.2+)
    private var processTapID: AudioObjectID = kAudioObjectUnknown
    private var tapAggregateDeviceID: AudioObjectID = kAudioObjectUnknown
    private var tapIOProcID: AudioDeviceIOProcID?
    private var tapSampleRate: Double = 48000.0
    private var tapChannels: Int = 2
    private var tapIsNonInterleaved: Bool = false

    // file playback
    private var fileTimer: DispatchSourceTimer?
    private let fileTimerQueue = DispatchQueue(label: "com.local.SwiftRVC.fileFeed", qos: .userInteractive)
    private var fileSamples: [Float] = []
    private var fileReadPos: Int = 0
    private var _filePlaying: Bool = true
    private var _fileLoop: Bool = true
    private var fileLock = os_unfair_lock_s()
    private var fileProgressCounter: Int = 0
	
    // pre-allocated buffers so coreaudio callbacks dont alloc
    private let captureScratchFrames: Int = 8192
    private var captureScratch: UnsafeMutablePointer<Float>
    private var monoScratch: UnsafeMutablePointer<Float>
    private var resampleScratch: UnsafeMutablePointer<Float>
    private let scratchFrames: Int = 8192
    private var outputScratch: UnsafeMutablePointer<Float>
    private var monitorScratch: UnsafeMutablePointer<Float>

    // cubic resampler state for 44.1k -> 48k
    private var inputDeviceSR: Double = 48000.0
    private var inputClientChannels: Int = 1
    private var resamplePhase: Double = 0.0
    private var histM2: Float = 0.0
    private var histM1: Float = 0.0
    private var hist0: Float = 0.0

    private var activeSourceMode: InputSourceMode = .mic
    private var inputCallbacks: Int = 0
    private var inputRenderErrorLogged: Bool = false
    private var signalSeen: Bool = false
    private var silenceWarned: Bool = false
    private var meterCounter: Int = 0

    // unix socket bridge
    private var sockFD: Int32 = -1
    private let blockSemaphore = DispatchSemaphore(value: 0)
    private let bridgeQueue = DispatchQueue(label: "com.local.SwiftRVC.audioBridge", qos: .userInteractive)
    private var isRunning: Bool = false
    private var bridgeRunning: Bool = false

    private var _blockSamples: Int = 7680
    private var _monitorEnabled: Bool = false
    private var _bypassDry: Bool = false
    private var paramLock = os_unfair_lock_s()
    
    // live wav output recorder (streams straight to disk)
    private let recordQueue = DispatchQueue(label: "com.local.SwiftRVC.wavRecord", qos: .utility)
    private var recordLock = os_unfair_lock_s()
    private var _isRecordingOutput: Bool = false
    private var outputRecordFile: AVAudioFile?
    private var outputRecordURL: URL?
    private var outputRecordFormat: AVAudioFormat?
    private var outputRecordedSamples: Int = 0
    private var outputRecordError: Error?
    private var outputRecordProgressTick: Int = 0

    var onInputLevel: ((Float) -> Void)?
    var onOutputLevel: ((Float) -> Void)?
    var onFileProgress: ((Int, Int, Bool) -> Void)?
    var onOutputRecordProgress: ((Double) -> Void)?
    var onLog: ((String, String) -> Void)?

    init() {
        captureScratch = .allocate(capacity: captureScratchFrames * 8)
        monoScratch = .allocate(capacity: captureScratchFrames)
        resampleScratch = .allocate(capacity: captureScratchFrames * 4)
        outputScratch = .allocate(capacity: scratchFrames)
        monitorScratch = .allocate(capacity: scratchFrames)
    }

    deinit {
        stop()
        captureScratch.deallocate()
        monoScratch.deallocate()
        resampleScratch.deallocate()
        outputScratch.deallocate()
        monitorScratch.deallocate()
    }

    private func log(_ tag: String, _ msg: String) {
        if let cb = onLog {
            DispatchQueue.main.async { cb(tag, msg) }
        }
    }

    var blockSamples: Int {
        get {
            os_unfair_lock_lock(&paramLock)
            let v = _blockSamples
            os_unfair_lock_unlock(&paramLock)
            return v
        }
        set {
            os_unfair_lock_lock(&paramLock)
            _blockSamples = max(480, newValue)
            os_unfair_lock_unlock(&paramLock)
        }
    }

    var monitorEnabled: Bool {
        get {
            os_unfair_lock_lock(&paramLock)
            let v = _monitorEnabled
            os_unfair_lock_unlock(&paramLock)
            return v
        }
        set {
            os_unfair_lock_lock(&paramLock)
            _monitorEnabled = newValue
            os_unfair_lock_unlock(&paramLock)
            if !newValue { monRing.reset() }
        }
    }

    var bypassDry: Bool {
        get {
            os_unfair_lock_lock(&paramLock)
            let v = _bypassDry
            os_unfair_lock_unlock(&paramLock)
            return v
        }
        set {
            os_unfair_lock_lock(&paramLock)
            _bypassDry = newValue
            os_unfair_lock_unlock(&paramLock)
        }
    }
    
		
    
	
    // file playback controls

    func setFilePlayback(playing: Bool) {
        os_unfair_lock_lock(&fileLock)
        _filePlaying = playing
        let pos = fileReadPos
        let total = fileSamples.count
        os_unfair_lock_unlock(&fileLock)
        if let cb = onFileProgress {
            DispatchQueue.main.async { cb(pos, total, playing) }
        }
    }

    func setFileLoop(_ loop: Bool) {
        os_unfair_lock_lock(&fileLock)
        _fileLoop = loop
        os_unfair_lock_unlock(&fileLock)
    }

    func seekFile(toFraction frac: Double) {
        os_unfair_lock_lock(&fileLock)
        let total = fileSamples.count
        let clamped = max(0.0, min(1.0, frac))
        fileReadPos = min(max(0, Int(Double(total) * clamped)), max(0, total - 1))
        let pos = fileReadPos
        let playing = _filePlaying
        os_unfair_lock_unlock(&fileLock)
        
        if let cb = onFileProgress {
            DispatchQueue.main.async { cb(pos, total, playing) }
        }
    }

    func updateLoadedFileSamples(_ samples: [Float]) {
        os_unfair_lock_lock(&fileLock)
        fileSamples = samples
        fileReadPos = 0
        let playing = _filePlaying
        os_unfair_lock_unlock(&fileLock)
        if let cb = onFileProgress {
            DispatchQueue.main.async { cb(0, samples.count, playing) }
        }
    }

    // start / stop

    func start(
        sourceMode: InputSourceMode,
        inputDevice: AudioDeviceInfo?,
        tapTarget: TappableAppInfo?,
        muteTappedApp: Bool,
        fileSamples: [Float],
        fileLoop: Bool,
        outputDevice: AudioDeviceInfo,
        monitorDevice: AudioDeviceInfo?,
        socketPath: String,
        blockSamples: Int
    ) throws {
        stop()

        self.activeSourceMode = sourceMode
        self.blockSamples = blockSamples
        inRing.reset()
        outRing.reset()
        monRing.reset()
        
        resamplePhase = 0.0
        histM2 = 0.0
        histM1 = 0.0
        hist0 = 0.0
        inputCallbacks = 0
        inputRenderErrorLogged = false
        signalSeen = false
        silenceWarned = false

        sockFD = try connectUnixSocket(path: socketPath)
        log("AUDIO", "Connected audio socket")

        do {
            let (outU, outHwSR, outCh) = try createOutputUnit(deviceID: outputDevice.id, isMonitor: false)
            outputUnit = outU
            log("AUDIO", "Output: \(outputDevice.name) @ \(Int(outHwSR)) Hz (\(outCh)ch)")

            if let monDev = monitorDevice, monitorEnabled {
                if monDev.id == outputDevice.id {
                    log("WARN", "Monitor matches output (\(monDev.name))")
                } else {
                    let (monU, monHwSR, monCh) = try createOutputUnit(deviceID: monDev.id, isMonitor: true)
                    monitorUnit = monU
                    log("AUDIO", "Monitor: \(monDev.name) @ \(Int(monHwSR)) Hz (\(monCh)ch)")
                }
            }
			
            switch sourceMode {
            case .mic:
                guard let inDev = inputDevice else {
                    throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "No microphone selected"])
                }
                let (inU, actualInSR, actualInCh) = try createInputUnit(deviceID: inDev.id)
                inputUnit = inU
                let note = abs(actualInSR - Self.targetSampleRate) >= 1.0 ? " -> 48kHz" : ""
                log("AUDIO", "Mic: \(inDev.name) @ \(Int(actualInSR)) Hz (\(actualInCh)ch)\(note)")

            case .systemApp:
                if #available(macOS 14.2, *) {
                    guard let target = tapTarget else {
                        throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "No app selected"])
                    }
                    try setupProcessTap(target: target, muteOriginal: muteTappedApp)
                } else {
                    throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "App capture requires macOS 14.2+"])
                }

            case .file:
                guard !fileSamples.isEmpty else {
                    throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "No file loaded"])
                }
                os_unfair_lock_lock(&fileLock)
                self.fileSamples = fileSamples
                self.fileReadPos = 0
                self._filePlaying = true
                self._fileLoop = fileLoop
                os_unfair_lock_unlock(&fileLock)
                let dur = Double(fileSamples.count) / Self.targetSampleRate
                log("AUDIO", String(format: "File ready (%.1fs @ 48kHz)", dur))
            }
            
            if let u = outputUnit {
                let st = AudioOutputUnitStart(u)
                if st != noErr { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Output start failed (\(st))"]) }
            }
            if let u = monitorUnit {
                let st = AudioOutputUnitStart(u)
                if st != noErr { log("WARN", "Monitor start failed (\(st))") }
            }
            if let u = inputUnit {
                let st = AudioOutputUnitStart(u)
                if st != noErr { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Mic start failed (\(st))"]) }
            }
            if tapAggregateDeviceID != kAudioObjectUnknown, let ioProc = tapIOProcID {
                let st = AudioDeviceStart(tapAggregateDeviceID, ioProc)
                if st != noErr {
                    throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "App tap start failed (\(st))"])
                }
            }
            if sourceMode == .file {
                startFileFeedTimer()
            }
        } catch {
            stop()
            throw error
        }

        isRunning = true
        bridgeRunning = true
        startBridgeLoop()
    }

    func updateMonitor(monitorDevice: AudioDeviceInfo?, outputDeviceID: AudioDeviceID, enabled: Bool) {
        monitorEnabled = enabled
        guard isRunning else { return }

        if let u = monitorUnit {
            AudioOutputUnitStop(u)
            AudioUnitUninitialize(u)
            AudioComponentInstanceDispose(u)
            monitorUnit = nil
        }
        monRing.reset()
        
        if enabled, let monDev = monitorDevice {
            if monDev.id == outputDeviceID {
                log("WARN", "Monitor matches output (\(monDev.name))")
                return
            }
            do {
                let (u, sr, ch) = try createOutputUnit(deviceID: monDev.id, isMonitor: true)
                monitorUnit = u
                let st = AudioOutputUnitStart(u)
                if st == noErr {
                    log("AUDIO", "Monitor on: \(monDev.name) @ \(Int(sr)) Hz (\(ch)ch)")
                } else {
                    log("ERR", "Monitor start failed (\(st))")
                }
            } catch {
                log("ERR", "Monitor error: \(error.localizedDescription)")
            }
        } else {
            log("AUDIO", "Monitor off")
        }
    }

    func playTestTone() {
        guard isRunning else {
            log("WARN", "Start audio first to play test tone")
            return
        }
        let n = 12000
        var tone = [Float](repeating: 0, count: n)
        for i in 0..<n {
            let env = sin(Float.pi * Float(i) / Float(n))
            tone[i] = 0.22 * env * sin(2.0 * Float.pi * 440.0 * Float(i) / 48000.0)
        }
        outRing.write(&tone, count: n, maxBacklog: 48000)
        if monitorEnabled {
            monRing.write(&tone, count: n, maxBacklog: 48000)
        }
        log("AUDIO", "Test tone -> output" + (monitorEnabled ? " + monitor" : ""))
    }

    func stop() {
        stopOutputRecording(completion: nil)
        let wasRunning = isRunning
        isRunning = false
        bridgeRunning = false
        blockSemaphore.signal()
        
        if let timer = fileTimer {
            timer.cancel()
            fileTimer = nil
        }
        teardownProcessTap()

        if let u = inputUnit {
            AudioOutputUnitStop(u)
            AudioUnitUninitialize(u)
            AudioComponentInstanceDispose(u)
            inputUnit = nil
        }
        if let u = outputUnit {
            AudioOutputUnitStop(u)
            AudioUnitUninitialize(u)
            AudioComponentInstanceDispose(u)
            outputUnit = nil
        }
        if let u = monitorUnit {
            AudioOutputUnitStop(u)
            AudioUnitUninitialize(u)
            AudioComponentInstanceDispose(u)
            monitorUnit = nil
        }
        if sockFD >= 0 {
            Darwin.shutdown(sockFD, SHUT_RDWR)
            Darwin.close(sockFD)
            sockFD = -1
        }
        
        inRing.reset()
        outRing.reset()
        monRing.reset()
        if wasRunning {
            log("AUDIO", "Stopped")
        }
    }
	
    
        
	
    
	
    // live wav output recorder

    func startOutputRecording(to url: URL) throws {
        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
        
        guard let fmt = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Self.targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "SwiftRVC", code: -30, userInfo: [NSLocalizedDescriptionKey: "Failed to build 48kHz mono format"])
        }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: Self.targetSampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]
        let wavFile = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)

        recordQueue.sync {
            self.outputRecordFile = wavFile
            self.outputRecordURL = url
            self.outputRecordFormat = fmt
            self.outputRecordedSamples = 0
            self.outputRecordError = nil
            self.outputRecordProgressTick = 0
        }
        os_unfair_lock_lock(&recordLock)
        _isRecordingOutput = true
        os_unfair_lock_unlock(&recordLock)
        log("AUDIO", "Recording -> \(url.lastPathComponent)")
    }

    private func tapOutputForRecording(_ samples: [Float], count: Int) {
        guard count > 0 else { return }
        os_unfair_lock_lock(&recordLock)
        let active = _isRecordingOutput
        os_unfair_lock_unlock(&recordLock)
        guard active else { return }

        let chunk = Array(samples.prefix(count))
        recordQueue.async { [weak self] in
            guard let self, let file = self.outputRecordFile, let fmt = self.outputRecordFormat else { return }
            guard let pcm = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(chunk.count)) else { return }
            pcm.frameLength = AVAudioFrameCount(chunk.count)
            if let dst = pcm.floatChannelData?[0] {
                for i in 0..<chunk.count {
                    dst[i] = max(-1.0, min(1.0, chunk[i]))
                }
            }
            do {
                try file.write(from: pcm)
                self.outputRecordedSamples += chunk.count
                let secs = Double(self.outputRecordedSamples) / Self.targetSampleRate
                self.outputRecordProgressTick += 1
                if self.outputRecordProgressTick >= 2 || chunk.count >= 4800 {
                    self.outputRecordProgressTick = 0
                    if let cb = self.onOutputRecordProgress {
                        DispatchQueue.main.async { cb(secs) }
                    }
                }
            } catch {
                if self.outputRecordError == nil {
                    self.outputRecordError = error
                    self.log("ERR", "WAV write error: \(error.localizedDescription)")
                }
            }
        }
    }

    func stopOutputRecording(completion: ((URL?, Double, Error?) -> Void)? = nil) {
        os_unfair_lock_lock(&recordLock)
        let wasRecording = _isRecordingOutput
        _isRecordingOutput = false
        os_unfair_lock_unlock(&recordLock)
        guard wasRecording else {
            if let completion {
                DispatchQueue.main.async { completion(nil, 0, nil) }
            }
            return
        }

        recordQueue.async { [weak self] in
            guard let self else { return }
            let url = self.outputRecordURL
            let dur = Double(self.outputRecordedSamples) / Self.targetSampleRate
            let err = self.outputRecordError
            
            // drop ref so avaudiofile flushes and closes the wav header
            self.outputRecordFile = nil
            self.outputRecordURL = nil
            self.outputRecordFormat = nil
            self.outputRecordedSamples = 0
            self.outputRecordError = nil
            if let completion {
                DispatchQueue.main.async { completion(url, dur, err) }
            }
        }
    }

    // offline export through the worker socket

    func convertFileOffline(
        samples: [Float],
        socketPath: String,
        blockSamples: Int,
        onProgress: ((Double) -> Void)? = nil
    ) throws -> [Float] {
        let fd = try connectUnixSocket(path: socketPath)
        defer {
            Darwin.shutdown(fd, SHUT_RDWR)
            Darwin.close(fd)
        }

        let bs = max(480, blockSamples)
        // pad one small tail block so lookahead doesnt clip the end
        let tailFlush = min(bs, 9600)
        let totalToSend = samples.count + tailFlush
        var output: [Float] = []
        output.reserveCapacity(totalToSend + bs)
        var inChunk = [Float](repeating: 0, count: bs)
        var outChunk = [Float](repeating: 0, count: bs)
        
        var pos = 0
        while pos < totalToSend {
            let n = min(bs, totalToSend - pos)
            for i in 0..<bs {
                let idx = pos + i
                inChunk[i] = idx < samples.count ? samples[idx] : 0
            }

            var byteLen = UInt32(bs * MemoryLayout<Float>.size).littleEndian
            let okH = withUnsafeBytes(of: &byteLen) { sendAll(fd: fd, ptr: $0.baseAddress!, count: 4) }
            let okB = inChunk.withUnsafeBytes { sendAll(fd: fd, ptr: $0.baseAddress!, count: bs * MemoryLayout<Float>.size) }
            guard okH && okB else {
                throw NSError(domain: "SwiftRVC", code: -10, userInfo: [NSLocalizedDescriptionKey: "Socket write failed"])
            }

            var respLenLE: UInt32 = 0
            guard withUnsafeMutableBytes(of: &respLenLE, { recvAll(fd: fd, ptr: $0.baseAddress!, count: 4) }) else {
                throw NSError(domain: "SwiftRVC", code: -11, userInfo: [NSLocalizedDescriptionKey: "Socket read failed"])
            }
            let respBytes = Int(UInt32(littleEndian: respLenLE))
            let respFloats = respBytes / MemoryLayout<Float>.size
            if outChunk.count < respFloats { outChunk = [Float](repeating: 0, count: respFloats) }
            guard outChunk.withUnsafeMutableBytes({ recvAll(fd: fd, ptr: $0.baseAddress!, count: respBytes) }) else {
                throw NSError(domain: "SwiftRVC", code: -12, userInfo: [NSLocalizedDescriptionKey: "Socket body read failed"])
            }

            output.append(contentsOf: outChunk.prefix(n))
            pos += n
            if let cb = onProgress {
                let frac = min(1.0, Double(pos) / Double(max(1, totalToSend)))
                DispatchQueue.main.async { cb(frac) }
            }
        }
        return output
    }
	
    
		
    
    
    // coreaudio process tap

    @available(macOS 14.2, *)
    private func setupProcessTap(target: TappableAppInfo, muteOriginal: Bool) throws {
        teardownProcessTap()

        let tapDesc: CATapDescription
        if target.isGlobalSystem {
            var exclude: [AudioObjectID] = []
            if let selfObj = AppAudioDiscovery.selfProcessObjectID() {
                exclude.append(selfObj)
            }
            tapDesc = CATapDescription(stereoGlobalTapButExcludeProcesses: exclude)
            tapDesc.name = "SwiftRVC System Tap"
        } else {
            let objs = AppAudioDiscovery.resolveProcessObjects(for: target)
            if objs.isEmpty {
                throw NSError(
                    domain: "SwiftRVC",
                    code: -20,
                    userInfo: [NSLocalizedDescriptionKey: "No active audio stream for \(target.name). Play audio there once and hit Refresh, or pick System Audio."]
                )
            }
            tapDesc = CATapDescription(stereoMixdownOfProcesses: objs)
            tapDesc.name = "SwiftRVC App Tap (\(target.name))"
        }

        tapDesc.uuid = UUID()
        tapDesc.isPrivate = true
        tapDesc.muteBehavior = muteOriginal ? .mutedWhenTapped : .unmuted
        
        var tapID: AudioObjectID = kAudioObjectUnknown
        let stTap = AudioHardwareCreateProcessTap(tapDesc, &tapID)
        guard stTap == noErr, tapID != kAudioObjectUnknown else {
            throw NSError(
                domain: "CoreAudio",
                code: Int(stTap),
                userInfo: [NSLocalizedDescriptionKey: "Process tap failed (\(stTap)). Check System Settings > Privacy & Security > System Audio Recording."]
            )
        }
        processTapID = tapID

        var tapFmt = AudioStreamBasicDescription()
        var fmtSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var fmtAddr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        _ = AudioObjectGetPropertyData(tapID, &fmtAddr, 0, nil, &fmtSize, &tapFmt)
        tapSampleRate = tapFmt.mSampleRate > 0 ? tapFmt.mSampleRate : 48000.0
        tapChannels = max(1, Int(tapFmt.mChannelsPerFrame))
        tapIsNonInterleaved = (tapFmt.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
        inputDeviceSR = tapSampleRate
		
        let aggUID = "com.local.SwiftRVC.tap.\(UUID().uuidString)"
        let aggDict: [String: Any] = [
            kAudioAggregateDeviceNameKey: "SwiftRVC Tap Device",
            kAudioAggregateDeviceUIDKey: aggUID,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceTapListKey: [
                [
                    kAudioSubTapUIDKey: tapDesc.uuid.uuidString,
                    kAudioSubTapDriftCompensationKey: true
                ]
            ]
        ]

        var aggID: AudioObjectID = kAudioObjectUnknown
        let stAgg = AudioHardwareCreateAggregateDevice(aggDict as CFDictionary, &aggID)
        guard stAgg == noErr, aggID != kAudioObjectUnknown else {
            teardownProcessTap()
            throw NSError(domain: "CoreAudio", code: Int(stAgg), userInfo: [NSLocalizedDescriptionKey: "Aggregate device failed (\(stAgg))"])
        }
        tapAggregateDeviceID = aggID

        var ioProc: AudioDeviceIOProcID?
        let stProc = AudioDeviceCreateIOProcID(
            aggID,
            CoreAudioBridgeEngine.tapIOProcCallback,
            Unmanaged.passUnretained(self).toOpaque(),
            &ioProc
        )
        guard stProc == noErr, let validProc = ioProc else {
            teardownProcessTap()
            throw NSError(domain: "CoreAudio", code: Int(stProc), userInfo: [NSLocalizedDescriptionKey: "Tap IOProc failed (\(stProc))"])
        }
        tapIOProcID = validProc

        let muteLabel = muteOriginal ? "muted" : "unmuted"
        log("AUDIO", "Tap: \(target.name) @ \(Int(tapSampleRate)) Hz (\(tapChannels)ch, \(muteLabel))")
    }

    private func teardownProcessTap() {
        if tapAggregateDeviceID != kAudioObjectUnknown {
            if let proc = tapIOProcID {
                _ = AudioDeviceStop(tapAggregateDeviceID, proc)
                _ = AudioDeviceDestroyIOProcID(tapAggregateDeviceID, proc)
                tapIOProcID = nil
            }
            _ = AudioHardwareDestroyAggregateDevice(tapAggregateDeviceID)
            tapAggregateDeviceID = kAudioObjectUnknown
        }
        if processTapID != kAudioObjectUnknown {
            if #available(macOS 14.2, *) {
                _ = AudioHardwareDestroyProcessTap(processTapID)
            }
            processTapID = kAudioObjectUnknown
        }
    }

    private static let tapIOProcCallback: AudioDeviceIOProc = { (
        _, _, inInputData, _, _, _, inClientData
    ) -> OSStatus in
        guard let inClientData else { return noErr }
        let engine = Unmanaged<CoreAudioBridgeEngine>.fromOpaque(inClientData).takeUnretainedValue()
        let abl = UnsafeMutableAudioBufferListPointer(UnsafeMutablePointer(mutating: inInputData))
        guard abl.count > 0 else { return noErr }

        var frames = 0
        if abl.count == 1 {
            // interleaved
            let buf = abl[0]
            let ch = max(1, Int(buf.mNumberChannels))
            guard let ptr = buf.mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            frames = min(Int(buf.mDataByteSize) / (MemoryLayout<Float>.size * ch), engine.captureScratchFrames)
            if ch == 1 {
                engine.monoScratch.update(from: ptr, count: frames)
            } else {
                let invCh = 1.0 / Float(ch)
                for i in 0..<frames {
                    var sum: Float = 0
                    let base = i * ch
                    for c in 0..<ch { sum += ptr[base + c] }
                    engine.monoScratch[i] = sum * invCh
                }
            }
        } else {
            // planar
            guard let firstPtr = abl[0].mData?.assumingMemoryBound(to: Float.self) else { return noErr }
            frames = min(Int(abl[0].mDataByteSize) / MemoryLayout<Float>.size, engine.captureScratchFrames)
            let nBufs = abl.count
            let invCh = 1.0 / Float(nBufs)
            for i in 0..<frames { engine.monoScratch[i] = firstPtr[i] * invCh }
            for b in 1..<nBufs {
                guard let p = abl[b].mData?.assumingMemoryBound(to: Float.self) else { continue }
                let nf = min(frames, Int(abl[b].mDataByteSize) / MemoryLayout<Float>.size)
                for i in 0..<nf { engine.monoScratch[i] += p[i] * invCh }
            }
        }

        if frames > 0 {
            engine.ingestMonoSamples(engine.monoScratch, frames: frames, sourceSR: engine.tapSampleRate)
        }
        return noErr
    }

    // file feeder (10ms ticks = 480 samples @ 48k)

    private func startFileFeedTimer() {
        let timer = DispatchSource.makeTimerSource(flags: .strict, queue: fileTimerQueue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(10), leeway: .milliseconds(1))
        let chunkFrames = 480
        var scratch = [Float](repeating: 0, count: chunkFrames)
        
        timer.setEventHandler { [weak self] in
            guard let self, self.isRunning else { return }

            os_unfair_lock_lock(&self.fileLock)
            let playing = self._filePlaying
            let loop = self._fileLoop
            let total = self.fileSamples.count
            var pos = self.fileReadPos
            var didEnd = false

            if playing && total > 0 {
                for i in 0..<chunkFrames {
                    if pos >= total {
                        if loop {
                            pos = 0
                        } else {
                            scratch[i] = 0
                            didEnd = true
                            continue
                        }
                    }
                    scratch[i] = self.fileSamples[pos]
                    pos += 1
                }
                self.fileReadPos = pos
                if didEnd { self._filePlaying = false }
            } else {
                for i in 0..<chunkFrames { scratch[i] = 0 }
            }
            let nowPlaying = self._filePlaying
            os_unfair_lock_unlock(&self.fileLock)

            scratch.withUnsafeMutableBufferPointer { buf in
                self.ingestMonoSamples(buf.baseAddress!, frames: chunkFrames, sourceSR: Self.targetSampleRate)
            }

            self.fileProgressCounter += 1
            if self.fileProgressCounter >= 6 || didEnd {
                self.fileProgressCounter = 0
                if let cb = self.onFileProgress {
                    DispatchQueue.main.async { cb(pos, total, nowPlaying) }
                }
            }
        }
        fileTimer = timer
        timer.resume()
    }

    // resample -> meter -> ring buffer

    private func ingestMonoSamples(_ monoIn: UnsafeMutablePointer<Float>, frames: Int, sourceSR: Double) {
        guard frames > 0 else { return }
        inputCallbacks += 1

        var outPtr: UnsafeMutablePointer<Float> = monoIn
        var outFrames: Int = frames

        if abs(sourceSR - Self.targetSampleRate) >= 1.0 {
            let step = sourceSR / Self.targetSampleRate
            var phase = resamplePhase
            let hM2 = histM2
            let hM1 = histM1
            let h0 = hist0
            let maxOut = captureScratchFrames * 4
            var k = 0
            let limit = Double(frames)

            @inline(__always)
            func sampleAt(_ idx: Int) -> Float {
                if idx >= 0 {
                    return monoIn[min(idx, frames - 1)]
                } else if idx == -1 {
                    return h0
                } else if idx == -2 {
                    return hM1
                } else {
                    return hM2
                }
            }
            
            // catmull-rom cubic interpolation
            while phase < limit && k < maxOut {
                let i1 = Int(phase) - 1
                let t = Float(phase - Double(Int(phase)))
                let p0 = sampleAt(i1 - 1)
                let p1 = sampleAt(i1)
                let p2 = sampleAt(i1 + 1)
                let p3 = sampleAt(i1 + 2)

                let t2 = t * t
                let t3 = t2 * t
                resampleScratch[k] = 0.5 * (
                    (2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
                )
                k += 1
                phase += step
            }

            if frames >= 3 {
                histM2 = monoIn[frames - 3]
                histM1 = monoIn[frames - 2]
                hist0 = monoIn[frames - 1]
            } else {
                for j in 0..<frames {
                    histM2 = histM1
                    histM1 = hist0
                    hist0 = monoIn[j]
                }
            }
            resamplePhase = phase - limit
            outPtr = resampleScratch
            outFrames = k
        }
		
        var rms: Float = 0
        if outFrames > 0 {
            vDSP_rmsqv(outPtr, 1, &rms, vDSP_Length(outFrames))
            let db = 20.0 * log10f(max(rms, 1e-6))

            if !signalSeen && rms > 1e-5 {
                signalSeen = true
                let srcLabel = activeSourceMode == .mic ? "Mic" : (activeSourceMode == .systemApp ? "App tap" : "File")
                log("AUDIO", String(format: "%@ signal ok (%.1f dB)", srcLabel, db))
            } else if !signalSeen && !silenceWarned && inputCallbacks == 80 && activeSourceMode != .file {
                silenceWarned = true
                if activeSourceMode == .systemApp {
                    log("WARN", "App tap is silent. Check System Settings > Privacy & Security > System Audio Recording")
                } else {
                    log("WARN", "Mic is silent. Check macOS Microphone permission or input gain")
                }
            }

            meterCounter += 1
            if meterCounter >= 3 {
                meterCounter = 0
                if let cb = onInputLevel {
                    DispatchQueue.main.async { cb(db) }
                }
            }
        }

        let avail = inRing.write(outPtr, count: outFrames, maxBacklog: blockSamples * 3)
        if avail >= blockSamples {
            blockSemaphore.signal()
        }
    }
    
	
        
    
	
    // unix socket ipc

    private func connectUnixSocket(path: String) throws -> Int32 {
        let fd = Darwin.socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            throw NSError(domain: "SwiftRVC", code: Int(errno), userInfo: [NSLocalizedDescriptionKey: "Socket create failed"])
        }
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathBytes = path.utf8CString
        let maxLen = MemoryLayout.size(ofValue: addr.sun_path)
        guard pathBytes.count <= maxLen else {
            Darwin.close(fd)
            throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "Socket path too long"])
        }
        withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            ptr.withMemoryRebound(to: CChar.self, capacity: maxLen) { dst in
                for i in 0..<pathBytes.count { dst[i] = pathBytes[i] }
            }
        }
        let res = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sPtr in
                Darwin.connect(fd, sPtr, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard res == 0 else {
            let err = errno
            Darwin.close(fd)
            throw NSError(domain: "SwiftRVC", code: Int(err), userInfo: [NSLocalizedDescriptionKey: "Socket connect failed (\(err))"])
        }
        return fd
    }

    private func sendAll(fd: Int32, ptr: UnsafeRawPointer, count: Int) -> Bool {
        var sent = 0
        while sent < count {
            let n = Darwin.send(fd, ptr.advanced(by: sent), count - sent, 0)
            if n <= 0 { return false }
            sent += n
        }
        return true
    }

    private func recvAll(fd: Int32, ptr: UnsafeMutableRawPointer, count: Int) -> Bool {
        var got = 0
        while got < count {
            let n = Darwin.recv(fd, ptr.advanced(by: got), count - got, 0)
            if n <= 0 { return false }
            got += n
        }
        return true
    }

    private func startBridgeLoop() {
        bridgeQueue.async { [weak self] in
            guard let self else { return }
            var inBuf = [Float](repeating: 0, count: 24000)
            var outBuf = [Float](repeating: 0, count: 24000)
            var blockCount = 0

            while self.bridgeRunning {
                let bs = self.blockSamples
                if inBuf.count < bs {
                    inBuf = [Float](repeating: 0, count: bs)
                    outBuf = [Float](repeating: 0, count: bs)
                }

                guard self.inRing.readExact(into: &inBuf, count: bs) else {
                    _ = self.blockSemaphore.wait(timeout: .now() + .milliseconds(20))
                    continue
                }

                blockCount += 1
                if blockCount == 1 {
                    self.log("AUDIO", "Streaming (\(bs) samples @ 48kHz)")
                }

                if self.bypassDry {
                    var rms: Float = 0
                    vDSP_rmsqv(&inBuf, 1, &rms, vDSP_Length(bs))
                    let outDB = 20.0 * log10f(max(rms, 1e-6))
                    if let cb = self.onOutputLevel {
                        DispatchQueue.main.async { cb(outDB) }
                    }
                    self.outRing.write(&inBuf, count: bs, maxBacklog: bs * 2)
                    if self.monitorEnabled {
                        self.monRing.write(&inBuf, count: bs, maxBacklog: bs * 2)
                    }
                    self.tapOutputForRecording(inBuf, count: bs)
                    continue
                }

                let fd = self.sockFD
                if fd < 0 { break }
                
                var byteLen = UInt32(bs * MemoryLayout<Float>.size).littleEndian
                let okSendHdr = withUnsafeBytes(of: &byteLen) { self.sendAll(fd: fd, ptr: $0.baseAddress!, count: 4) }
                if !okSendHdr {
                    self.log("ERR", "Audio socket send failed")
                    break
                }
                let okSendBody = inBuf.withUnsafeBytes { self.sendAll(fd: fd, ptr: $0.baseAddress!, count: bs * MemoryLayout<Float>.size) }
                if !okSendBody { break }

                var respLenLE: UInt32 = 0
                let okRecvHdr = withUnsafeMutableBytes(of: &respLenLE) { self.recvAll(fd: fd, ptr: $0.baseAddress!, count: 4) }
                if !okRecvHdr {
                    self.log("ERR", "Audio socket read failed")
                    break
                }

                let respBytes = Int(UInt32(littleEndian: respLenLE))
                let respFloats = respBytes / MemoryLayout<Float>.size
                if outBuf.count < respFloats { outBuf = [Float](repeating: 0, count: respFloats) }
                let okRecvBody = outBuf.withUnsafeMutableBytes { self.recvAll(fd: fd, ptr: $0.baseAddress!, count: respBytes) }
                if !okRecvBody { break }

                var outRms: Float = 0
                vDSP_rmsqv(&outBuf, 1, &outRms, vDSP_Length(respFloats))
                let outDB = 20.0 * log10f(max(outRms, 1e-6))
                if let cb = self.onOutputLevel {
                    DispatchQueue.main.async { cb(outDB) }
                }

                self.outRing.write(&outBuf, count: respFloats, maxBacklog: bs * 2)
                if self.monitorEnabled {
                    self.monRing.write(&outBuf, count: respFloats, maxBacklog: bs * 2)
                }
                self.tapOutputForRecording(outBuf, count: respFloats)
            }
        }
    }
	
    
    
		
    
	
    
    // coreaudio auhal units

    private func newHALAudioUnit() throws -> AudioUnit {
        var desc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        guard let comp = AudioComponentFindNext(nil, &desc) else {
            throw NSError(domain: "CoreAudio", code: -1, userInfo: [NSLocalizedDescriptionKey: "HALOutput component not found"])
        }
        var unit: AudioUnit?
        let st = AudioComponentInstanceNew(comp, &unit)
        guard st == noErr, let u = unit else {
            throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "AUHAL init failed (\(st))"])
        }
        return u
    }

    private func createInputUnit(deviceID: AudioDeviceID) throws -> (AudioUnit, Double, Int) {
        let u = try newHALAudioUnit()

        var enableIO: UInt32 = 1
        var disableIO: UInt32 = 0
        var st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &enableIO, 4)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Enable input failed (\(st))"]) }
        st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &disableIO, 4)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Disable output failed (\(st))"]) }

        var devID = deviceID
        st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &devID, UInt32(MemoryLayout<AudioDeviceID>.size))
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Set input device failed (\(st))"]) }
        
        // open at native hw sample rate, otherwise 44.1k mics return -10863 on render
        var hwFmt = AudioStreamBasicDescription()
        var fmtSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        _ = AudioUnitGetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1, &hwFmt, &fmtSize)
        let nativeSR = hwFmt.mSampleRate > 0 ? hwFmt.mSampleRate : 48000.0
        let hwCh = max(1, min(8, Int(hwFmt.mChannelsPerFrame)))
        inputDeviceSR = nativeSR

        var clientFmt = AudioStreamBasicDescription(
            mSampleRate: nativeSR,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )
        st = AudioUnitSetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &clientFmt, fmtSize)
        if st == noErr {
            inputClientChannels = 1
        } else {
            clientFmt.mChannelsPerFrame = UInt32(hwCh)
            clientFmt.mBytesPerFrame = UInt32(4 * hwCh)
            clientFmt.mBytesPerPacket = clientFmt.mBytesPerFrame
            st = AudioUnitSetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &clientFmt, fmtSize)
            guard st == noErr else {
                throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Set input format failed (\(st))"])
            }
            inputClientChannels = hwCh
        }

        var cb = AURenderCallbackStruct(
            inputProc: CoreAudioBridgeEngine.inputCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &cb, UInt32(MemoryLayout<AURenderCallbackStruct>.size))
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Set input callback failed (\(st))"]) }

        st = AudioUnitInitialize(u)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Input unit init failed (\(st))"]) }
        return (u, nativeSR, inputClientChannels)
    }

    private func createOutputUnit(deviceID: AudioDeviceID, isMonitor: Bool) throws -> (AudioUnit, Double, Int) {
        let u = try newHALAudioUnit()

        var enableIO: UInt32 = 1
        var disableIO: UInt32 = 0
        var st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &disableIO, 4)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Disable input failed (\(st))"]) }
        st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &enableIO, 4)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Enable output failed (\(st))"]) }

        var devID = deviceID
        st = AudioUnitSetProperty(u, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &devID, UInt32(MemoryLayout<AudioDeviceID>.size))
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Set output device failed (\(st))"]) }
        
        var hwFmt = AudioStreamBasicDescription()
        var fmtSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        _ = AudioUnitGetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &hwFmt, &fmtSize)
        let hwSR = hwFmt.mSampleRate > 0 ? hwFmt.mSampleRate : 48000.0
        let hwCh = max(1, Int(hwFmt.mChannelsPerFrame))

        var outFmt = AudioStreamBasicDescription(
            mSampleRate: Self.targetSampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: UInt32(hwCh),
            mBitsPerChannel: 32,
            mReserved: 0
        )
        st = AudioUnitSetProperty(u, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &outFmt, fmtSize)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Set output format failed (\(st))"]) }

        var cb = AURenderCallbackStruct(
            inputProc: isMonitor ? CoreAudioBridgeEngine.monitorOutputCallback : CoreAudioBridgeEngine.mainOutputCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        st = AudioUnitSetProperty(u, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &cb, UInt32(MemoryLayout<AURenderCallbackStruct>.size))
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Set render callback failed (\(st))"]) }

        st = AudioUnitInitialize(u)
        guard st == noErr else { throw NSError(domain: "CoreAudio", code: Int(st), userInfo: [NSLocalizedDescriptionKey: "Output unit init failed (\(st))"]) }
        return (u, hwSR, hwCh)
    }

    // audio render callbacks

    private static let inputCallback: AURenderCallback = { (
        inRefCon, ioActionFlags, inTimeStamp, _, inNumberFrames, _
    ) -> OSStatus in
        let engine = Unmanaged<CoreAudioBridgeEngine>.fromOpaque(inRefCon).takeUnretainedValue()
        guard let unit = engine.inputUnit else { return noErr }
        let frames = min(Int(inNumberFrames), engine.captureScratchFrames)
        let ch = max(1, engine.inputClientChannels)
        let byteSize = UInt32(frames * ch * MemoryLayout<Float>.size)

        var abl = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: UInt32(ch),
                mDataByteSize: byteSize,
                mData: UnsafeMutableRawPointer(engine.captureScratch)
            )
        )

        let st = AudioUnitRender(unit, ioActionFlags, inTimeStamp, 1, UInt32(frames), &abl)
        if st != noErr {
            if !engine.inputRenderErrorLogged {
                engine.inputRenderErrorLogged = true
                engine.log("ERR", "Mic render failed (\(st))")
            }
            return st
        }
        guard frames > 0 else { return noErr }

        var monoIn: UnsafeMutablePointer<Float> = engine.captureScratch
        if ch > 1 {
            let invCh = 1.0 / Float(ch)
            for i in 0..<frames {
                var sum: Float = 0
                let base = i * ch
                for c in 0..<ch { sum += engine.captureScratch[base + c] }
                engine.monoScratch[i] = sum * invCh
            }
            monoIn = engine.monoScratch
        }

        engine.ingestMonoSamples(monoIn, frames: frames, sourceSR: engine.inputDeviceSR)
        return noErr
    }

    private static let mainOutputCallback: AURenderCallback = { (
        inRefCon, _, _, _, inNumberFrames, ioData
    ) -> OSStatus in
        let engine = Unmanaged<CoreAudioBridgeEngine>.fromOpaque(inRefCon).takeUnretainedValue()
        return engine.fillOutputBuffers(ioData: ioData, frames: Int(inNumberFrames), ring: engine.outRing, scratch: engine.outputScratch)
    }

    private static let monitorOutputCallback: AURenderCallback = { (
        inRefCon, _, _, _, inNumberFrames, ioData
    ) -> OSStatus in
        let engine = Unmanaged<CoreAudioBridgeEngine>.fromOpaque(inRefCon).takeUnretainedValue()
        return engine.fillOutputBuffers(ioData: ioData, frames: Int(inNumberFrames), ring: engine.monRing, scratch: engine.monitorScratch)
    }

    private func fillOutputBuffers(
        ioData: UnsafeMutablePointer<AudioBufferList>?,
        frames: Int,
        ring: FloatRingBuffer,
        scratch: UnsafeMutablePointer<Float>
    ) -> OSStatus {
        guard let ioData else { return noErr }
        let n = min(frames, scratchFrames)
        let got = ring.read(into: scratch, count: n)
        if got < n {
            scratch.advanced(by: got).initialize(repeating: 0, count: n - got)
        }
        let abl = UnsafeMutableAudioBufferListPointer(ioData)
        for buf in abl {
            guard let dst = buf.mData?.assumingMemoryBound(to: Float.self) else { continue }
            dst.update(from: scratch, count: n)
            if frames > n {
                dst.advanced(by: n).initialize(repeating: 0, count: frames - n)
            }
        }
        return noErr
    }
}
