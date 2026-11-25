import AppKit
import AVFoundation
import CoreAudio
import Foundation
import UniformTypeIdentifiers

struct LogEntry: Identifiable {
    let id: Int
    let timestamp: String
    let tag: String
    let message: String
}

enum VoiceEngineType: String, CaseIterable, Identifiable {
    case rvc = "rvc"
    case beatrice = "beatrice"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .rvc: return "RVC v2"
        case .beatrice: return "Beatrice v2"
        }
    }

    var badge: String {
        switch self {
        case .rvc: return "MLX"
        case .beatrice: return "NEON"
        }
    }
}

struct BeatriceSpeakerMeta: Identifiable, Hashable {
    let id: Int
    let name: String
    let label: String
    let averagePitchHz: Double
}

struct VoiceModelMeta: Identifiable, Hashable {
    let id: String
    let engine: VoiceEngineType
    let name: String
    let path: String
    let indexPath: String?
    let sampleRate: Int
    let version: String
    let f0: Int
    let speakers: [BeatriceSpeakerMeta]

    var badgeText: String {
        if engine == .beatrice {
            return "\(version) · \(speakers.count) voice\(speakers.count == 1 ? "" : "s")"
        }
        let idx = indexPath != nil ? " · index" : ""
        return "\(version) · \(sampleRate / 1000)kHz\(idx)"
    }
}

// keep field names stable so existing model_settings.json keeps loading cleanly
struct SavedModelSettings: Codable {
    var pitchShift: Double
    var formantShift: Double
    var pitchFloorHz: Double
    var voicePolish: Bool
    var gateDB: Double
    var indexRate: Double
    var protect: Double
    var noiseScale: Double
    var intonation: Double
    var rmsMix: Double
    var outputGain: Double
    var speakerID: Int
}

enum EfficiencyPreset: String, CaseIterable, Identifiable {
    case ultraPerformance = "ultra_performance"
    case performance = "performance"
    case balanced = "balanced"
    case quality = "quality"
    case ultraQuality = "ultra_quality"

    var id: String { rawValue }

    var shortTitle: String {
        switch self {
        case .ultraPerformance: return "Ultra Perf"
        case .performance: return "Perf"
        case .balanced: return "Balanced"
        case .quality: return "Quality"
        case .ultraQuality: return "Ultra Qual"
        }
    }

    var title: String {
        switch self {
        case .ultraPerformance: return "Ultra Performance"
        case .performance: return "Performance"
        case .balanced: return "Balanced"
        case .quality: return "Quality"
        case .ultraQuality: return "Ultra Quality"
        }
    }

    func summary(for engine: VoiceEngineType) -> String {
        if engine == .beatrice {
            switch self {
            case .ultraPerformance: return "20ms · vq 0 · ~60ms lat"
            case .performance: return "30ms · vq 1 · ~70ms lat"
            case .balanced: return "40ms · vq 2 · ~82ms lat"
            case .quality: return "60ms · vq 4 · ~105ms lat"
            case .ultraQuality: return "80ms · vq 6 · ~125ms lat"
            }
        }
        switch self {
        case .ultraPerformance: return "80ms · 10ms la · YIN · 8-bit"
        case .performance: return "100ms · 20ms la · YIN · 8-bit"
        case .balanced: return "160ms · 35ms la · RMVPE fp16"
        case .quality: return "240ms · 55ms la · RMVPE + index"
        case .ultraQuality: return "300ms · 80ms la · RMVPE fp32 + index"
        }
    }
}

private final class OutputPreviewPlayerDelegate: NSObject, AVAudioPlayerDelegate {
    var onFinish: (() -> Void)?
    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        DispatchQueue.main.async { [weak self] in self?.onFinish?() }
    }
}
	
@MainActor
final class VoiceChangerController: ObservableObject {
    // audio devices & source mode
    @Published var inputSourceMode: InputSourceMode = .mic
    @Published var inputDevices: [AudioDeviceInfo] = []
    @Published var outputDevices: [AudioDeviceInfo] = []
    @Published var selectedInputID: AudioDeviceID = 0
    @Published var selectedOutputID: AudioDeviceID = 0
    @Published var selectedMonitorID: AudioDeviceID = 0
    @Published var monitorEnabled: Bool = false {
        didSet { syncMonitorOutput() }
    }

    // process tap
    @Published var tappableApps: [TappableAppInfo] = []
    @Published var selectedTapAppID: String = TappableAppInfo.allSystemID
    @Published var muteTappedApp: Bool = false

    // file source
    @Published var loadedFileName: String = ""
    @Published var loadedFileSamples: [Float] = []
    @Published var fileIsPlaying: Bool = true
    @Published var fileLoop: Bool = true
    @Published var filePositionSamples: Int = 0
    @Published var fileTotalSamples: Int = 0
    @Published var isRecordingSample: Bool = false
    @Published var recordingSeconds: Double = 0.0
    @Published var isExportingWav: Bool = false
    @Published var exportProgress: Double = 0.0

    // live output recorder
    @Published var isRecordingOutput: Bool = false
    @Published var outputRecordSeconds: Double = 0.0
    @Published var lastRecordedOutputURL: URL? = nil
    @Published var lastRecordedOutputDuration: Double = 0.0
    @Published var isPreviewingRecordedOutput: Bool = false
    
    var fileProgressFraction: Double {
        guard fileTotalSamples > 0 else { return 0.0 }
        return min(1.0, max(0.0, Double(filePositionSamples) / Double(fileTotalSamples)))
    }

    var fileCurrentTimeText: String {
        Self.formatDuration(Double(filePositionSamples) / 48000.0)
    }

    var fileTotalTimeText: String {
        Self.formatDuration(Double(fileTotalSamples) / 48000.0)
    }

    static func formatDuration(_ sec: Double) -> String {
        let s = max(0, Int(sec.rounded(.down)))
        return String(format: "%02d:%02d", s / 60, s % 60)
    }

    // engine & model
    @Published var activeEngine: VoiceEngineType = .rvc
    @Published var beatriceAvailable: Bool = false
    @Published var rvcModels: [VoiceModelMeta] = []
    @Published var beatriceModels: [VoiceModelMeta] = []
    @Published var selectedModelPath: String = ""
    @Published var selectedSpeakerID: Int = 3
    @Published var preset: EfficiencyPreset = .balanced

    // tuning params
    @Published var pitchShift: Double = 0.0
    @Published var formantShift: Double = 0.0
    @Published var pitchFloorHz: Double = 0.0
    @Published var voicePolish: Bool = true
    @Published var gateDB: Double = -40.0
    @Published var indexRate: Double = 0.35
    @Published var protect: Double = 0.30
    @Published var noiseScale: Double = 0.32
    @Published var intonation: Double = 1.0
    @Published var rmsMix: Double = 0.35
    @Published var outputGain: Double = 1.0
    @Published var bypass: Bool = false
    @Published var showMoreControls: Bool = false
    @Published var showLogPanel: Bool = true

    // runtime state
    @Published var workerReady: Bool = false
    @Published var isLoadingModel: Bool = false
    @Published var isStreaming: Bool = false
    @Published var statusMessage: String = "Starting..."
    @Published var errorMessage: String? = nil

    @Published var inferMS: Double = 0.0
    @Published var totalLatencyMS: Double = 0.0
    @Published var rtf: Double = 0.0
    @Published var ramMB: Double = 0.0
    @Published var isGated: Bool = true
    @Published var inputLevelDB: Float = -80.0
    @Published var outputLevelDB: Float = -80.0
    @Published var currentBlockSamples: Int = 7680
    @Published var indexLoaded: Bool = false
    @Published var indexRamMB: Double = 0.0

    @Published var logs: [LogEntry] = []
    private var nextLogID: Int = 1
    private var lastGatedState: Bool? = nil
    private var lastTelemetryLogTime: Date = .distantPast
    private let timeFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "HH:mm:ss"
        return df
    }()

    private var savedSettingsMap: [String: SavedModelSettings] = [:]
    private var suppressSettingsSave: Bool = false
    private let settingsFileURL: URL

    var activeModels: [VoiceModelMeta] {
        activeEngine == .beatrice ? beatriceModels : rvcModels
    }

    var selectedModel: VoiceModelMeta? {
        activeModels.first { $0.path == selectedModelPath }
    }

    var hasVirtualCable: Bool {
        outputDevices.contains { $0.isVirtualCable }
    }
    
    private let audioEngine = CoreAudioBridgeEngine()
    private var sampleRecorder: AVAudioRecorder?
    private var recordTimer: Timer?
    private var loadedFileURL: URL?
    private let sampleRecordURL = URL(fileURLWithPath: "/tmp/swiftrvc_mic_sample.wav")
    private let recordingsDirURL: URL
    private var pendingStartOutputRecording: Bool = false
    private var outputPreviewPlayer: AVAudioPlayer?
    private var outputPreviewDelegate: OutputPreviewPlayerDelegate?
    private let fileTimestampFormatter: DateFormatter = {
        let df = DateFormatter()
        df.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        return df
    }()

    private var workerProcess: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var socketPath: String = ""
    private var audioSocketReady: Bool = false
    private var reqCounter: Int = 0
    private var pendingCallbacks: [Int: ([String: Any]) -> Void] = [:]
    private let projectRoot: String
    private var paramDebounceTask: Task<Void, Never>?

    init() {
        let root = Self.resolveProjectRoot()
        self.projectRoot = root
        self.settingsFileURL = URL(fileURLWithPath: "\(root)/model_settings.json")
        self.recordingsDirURL = URL(fileURLWithPath: "\(root)/recordings", isDirectory: true)
        self.socketPath = "/tmp/swiftrvc_\(ProcessInfo.processInfo.processIdentifier).sock"
        
        if let savedPresetRaw = UserDefaults.standard.string(forKey: "SwiftRVC.lastPreset"),
           let savedPreset = EfficiencyPreset(rawValue: savedPresetRaw) {
            self.preset = savedPreset
        }

        loadSavedSettingsStore()

        audioEngine.onInputLevel = { [weak self] db in
            self?.inputLevelDB = db
        }
        audioEngine.onOutputLevel = { [weak self] db in
            self?.outputLevelDB = db
        }
        audioEngine.onFileProgress = { [weak self] pos, total, playing in
            self?.filePositionSamples = pos
            self?.fileTotalSamples = total
            self?.fileIsPlaying = playing
        }
        audioEngine.onOutputRecordProgress = { [weak self] secs in
            self?.outputRecordSeconds = secs
        }
        audioEngine.onLog = { [weak self] tag, msg in
            self?.appendLog(tag, msg)
        }

        appendLog("INFO", "SwiftRVC started")
        refreshDevices()
        installDeviceChangeListener()
        startWorker()
    }

    static func resolveProjectRoot() -> String {
        let fm = FileManager.default
        let bundleParent = Bundle.main.bundleURL.deletingLastPathComponent().path
        if fm.fileExists(atPath: "\(bundleParent)/engine/worker.py") {
            return bundleParent
        }
        if let res = Bundle.main.resourcePath, fm.fileExists(atPath: "\(res)/engine/worker.py") {
            return res
        }
        let cwd = fm.currentDirectoryPath
        if fm.fileExists(atPath: "\(cwd)/engine/worker.py") {
            return cwd
        }
        let parent = URL(fileURLWithPath: cwd).deletingLastPathComponent().path
        if fm.fileExists(atPath: "\(parent)/engine/worker.py") {
            return parent
        }
        return bundleParent
    }

    // per-model settings persistence

    private func loadSavedSettingsStore() {
        guard let data = try? Data(contentsOf: settingsFileURL),
              let decoded = try? JSONDecoder().decode([String: SavedModelSettings].self, from: data) else {
            return
        }
        savedSettingsMap = decoded
    }

    private func persistSettingsStore() {
        guard let data = try? JSONEncoder().encode(savedSettingsMap) else { return }
        try? data.write(to: settingsFileURL, options: .atomic)
    }

    private func saveCurrentModelSettings() {
        guard !suppressSettingsSave, !selectedModelPath.isEmpty else { return }
        let snap = SavedModelSettings(
            pitchShift: pitchShift,
            formantShift: formantShift,
            pitchFloorHz: pitchFloorHz,
            voicePolish: voicePolish,
            gateDB: gateDB,
            indexRate: indexRate,
            protect: protect,
            noiseScale: noiseScale,
            intonation: intonation,
            rmsMix: rmsMix,
            outputGain: outputGain,
            speakerID: selectedSpeakerID
        )
        savedSettingsMap[selectedModelPath] = snap
        persistSettingsStore()
    }

    @discardableResult
    private func restoreSettingsIfSaved(for modelPath: String, modelName: String) -> Bool {
        guard let s = savedSettingsMap[modelPath] else { return false }
        suppressSettingsSave = true
        pitchShift = s.pitchShift
        formantShift = s.formantShift
        pitchFloorHz = s.pitchFloorHz
        voicePolish = s.voicePolish
        gateDB = s.gateDB
        indexRate = s.indexRate
        protect = s.protect
        noiseScale = s.noiseScale
        intonation = s.intonation
        rmsMix = s.rmsMix
        outputGain = s.outputGain
        selectedSpeakerID = s.speakerID
        suppressSettingsSave = false
        appendLog("INFO", "Loaded saved settings for \(modelName)")
        return true
    }

    // logs

    func appendLog(_ tag: String, _ message: String) {
        let ts = timeFormatter.string(from: Date())
        let entry = LogEntry(id: nextLogID, timestamp: ts, tag: tag, message: message)
        nextLogID += 1
        logs.append(entry)
        if logs.count > 250 {
            logs.removeFirst(logs.count - 250)
        }
    }

    func clearLogs() {
        logs.removeAll()
        appendLog("INFO", "Cleared")
    }

    func copyLogs() {
        let text = logs.map { "[\($0.timestamp)] [\($0.tag)] \($0.message)" }.joined(separator: "\n")
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
	
    
		
    
    // devices & routing

    func refreshDevices() {
        let all = AudioDeviceDiscovery.allDevices()
        inputDevices = all.filter { $0.inputChannels > 0 }
        outputDevices = all.filter { $0.outputChannels > 0 }

        if !inputDevices.contains(where: { $0.id == selectedInputID }) {
            let defIn = AudioDeviceDiscovery.defaultDeviceID(isInput: true)
            selectedInputID = inputDevices.first(where: { $0.id == defIn })?.id
                ?? inputDevices.first?.id ?? 0
        }

        if !outputDevices.contains(where: { $0.id == selectedOutputID }) {
            if let bh = outputDevices.first(where: { $0.isVirtualCable }) {
                selectedOutputID = bh.id
            } else {
                let defOut = AudioDeviceDiscovery.defaultDeviceID(isInput: false)
                selectedOutputID = outputDevices.first(where: { $0.id == defOut })?.id
                    ?? outputDevices.first?.id ?? 0
            }
        }

        if !outputDevices.contains(where: { $0.id == selectedMonitorID }) {
            let defOut = AudioDeviceDiscovery.defaultDeviceID(isInput: false)
            selectedMonitorID = outputDevices.first(where: { $0.id == defOut && !$0.isVirtualCable })?.id
                ?? outputDevices.first(where: { !$0.isVirtualCable })?.id
                ?? outputDevices.first?.id ?? 0
        }

        refreshTappableApps(logCount: false)
        appendLog("AUDIO", "Devices: \(inputDevices.count) in, \(outputDevices.count) out, \(max(0, tappableApps.count - 1)) apps")
    }

    func refreshTappableApps(logCount: Bool = true) {
        tappableApps = AppAudioDiscovery.tappableSources()
        if !tappableApps.contains(where: { $0.id == selectedTapAppID }) {
            selectedTapAppID = TappableAppInfo.allSystemID
        }
        if logCount {
            let playing = tappableApps.filter { $0.isPlaying }.count
            appendLog("AUDIO", "Apps: \(max(0, tappableApps.count - 1)) (\(playing) playing)")
        }
    }

    func switchInputSourceMode(to mode: InputSourceMode) {
        guard mode != inputSourceMode else { return }
        if isRecordingSample {
            finishMicSampleRecording(autoSwitchToFile: false)
        }
        inputSourceMode = mode
        errorMessage = nil
        if mode == .systemApp {
            refreshTappableApps(logCount: false)
        }
        appendLog("AUDIO", "Source -> \(mode.title)")
        
        if isStreaming {
            if mode == .file && loadedFileSamples.isEmpty {
                stopStreaming()
                statusMessage = "No file loaded"
            } else {
                stopStreaming()
                startStreaming()
            }
        }
    }

    func syncMonitorOutput() {
        let monDev = outputDevices.first { $0.id == selectedMonitorID }
        audioEngine.updateMonitor(
            monitorDevice: monDev,
            outputDeviceID: selectedOutputID,
            enabled: monitorEnabled
        )
    }

    func playTestTone() {
        audioEngine.playTestTone()
    }

    private func installDeviceChangeListener() {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &addr,
            DispatchQueue.main
        ) { [weak self] _, _ in
            self?.refreshDevices()
        }
    }
    
	
        
    
    
	
    // worker process

    func startWorker() {
        stopWorker()
        errorMessage = nil
        workerReady = false
        audioSocketReady = false
        statusMessage = "Starting..."
        appendLog("MLX", "Starting worker (\(preset.title))")
        
        let pythonPath = "\(projectRoot)/.venv/bin/python"
        let workerScript = "\(projectRoot)/engine/worker.py"
        let baseModelsDir = "\(projectRoot)/base_models"
        let modelsDir = "\(projectRoot)/models"
        let beatriceModelsDir = "\(projectRoot)/beatrice_models"
        let beatriceDylib = "\(projectRoot)/beatrice/libbeatrice_bridge.dylib"

        guard FileManager.default.isExecutableFile(atPath: pythonPath) else {
            let msg = "Python venv missing. Run ./setup.sh first."
            errorMessage = msg
            statusMessage = "Missing .venv"
            appendLog("ERR", msg)
            return
        }

        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: pythonPath)
        proc.arguments = [
            "-u", workerScript,
            "--base-models", baseModelsDir,
            "--models-dir", modelsDir,
            "--beatrice-models", beatriceModelsDir,
            "--beatrice-dylib", beatriceDylib,
            "--socket", socketPath,
            "--preset", preset.rawValue
        ]
        proc.currentDirectoryURL = URL(fileURLWithPath: projectRoot)

        let inPipe = Pipe()
        let outPipe = Pipe()
        let errPipe = Pipe()
        proc.standardInput = inPipe
        proc.standardOutput = outPipe
        proc.standardError = errPipe

        self.stdinPipe = inPipe
        self.stdoutPipe = outPipe
        self.workerProcess = proc

        let outHandle = outPipe.fileHandleForReading
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            var buffer = Data()
            while true {
                let chunk = outHandle.availableData
                if chunk.isEmpty { break }
                buffer.append(chunk)
                while let nl = buffer.firstIndex(of: 0x0A) {
                    let lineData = buffer.subdata(in: 0..<nl)
                    buffer.removeSubrange(0...nl)
                    if let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] {
                        Task { @MainActor [weak self] in
                            self?.handleWorkerEvent(obj)
                        }
                    }
                }
            }
            Task { @MainActor [weak self] in
                guard let self else { return }
                if self.workerProcess === proc {
                    self.workerReady = false
                    self.appendLog("WARN", "Worker exited")
                    if self.isStreaming { self.stopStreaming() }
                }
            }
        }

        let errHandle = errPipe.fileHandleForReading
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let data = errHandle.readDataToEndOfFile()
            if !data.isEmpty, let msg = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !msg.isEmpty {
                Task { @MainActor [weak self] in
                    self?.appendLog("ERR", "Worker stderr: \(String(msg.suffix(220)))")
                    if self?.errorMessage == nil && !(self?.workerReady ?? false) {
                        self?.errorMessage = String(msg.suffix(240))
                    }
                }
            }
        }

        do {
            try proc.run()
        } catch {
            errorMessage = "Worker launch failed: \(error.localizedDescription)"
            statusMessage = "Launch failed"
            appendLog("ERR", errorMessage ?? "")
        }
    }

    func stopWorker() {
        stopPreviewLastRecording()
        if isRecordingSample {
            recordTimer?.invalidate()
            recordTimer = nil
            sampleRecorder?.stop()
            sampleRecorder = nil
            isRecordingSample = false
        }
        stopStreaming()
        if let proc = workerProcess, proc.isRunning {
            sendCommand(["cmd": "quit"])
            usleep(60_000)
            if proc.isRunning { proc.terminate() }
        }
        workerProcess = nil
        stdinPipe = nil
        stdoutPipe = nil
        workerReady = false
        audioSocketReady = false
        try? FileManager.default.removeItem(atPath: socketPath)
    }

    private func sendCommand(_ payload: [String: Any], completion: (([String: Any]) -> Void)? = nil) {
        guard let pipe = stdinPipe else { return }
        reqCounter += 1
        let reqID = reqCounter
        var body = payload
        body["id"] = reqID
        if let cb = completion {
            pendingCallbacks[reqID] = cb
        }
        guard let data = try? JSONSerialization.data(withJSONObject: body) else { return }
        var line = data
        line.append(0x0A)
        pipe.fileHandleForWriting.write(line)
    }

    private func handleWorkerEvent(_ ev: [String: Any]) {
        guard let type = ev["type"] as? String else { return }
        switch type {
        case "audio_socket_ready":
            audioSocketReady = true
            appendLog("MLX", "Socket ready")
        case "ready":
            workerReady = true
            statusMessage = "Ready"
            beatriceAvailable = ev["beatrice_available"] as? Bool ?? false
            if let list = ev["models"] as? [[String: Any]] {
                applyRVCModelList(list)
            }
            if let blist = ev["beatrice_models"] as? [[String: Any]] {
                applyBeatriceModelList(blist)
            }
            appendLog("MLX", "Ready (\(rvcModels.count) RVC, \(beatriceModels.count) Beatrice)")
            if let info = ev["info"] as? [String: Any] {
                applyModelInfo(info, syncIndexRate: true)
            }
            if selectedModelPath.isEmpty {
                let savedPath = UserDefaults.standard.string(forKey: "SwiftRVC.lastModelPath") ?? ""
                if let remembered = activeModels.first(where: { $0.path == savedPath }) {
                    selectedModelPath = remembered.path
                    loadSelectedModel()
                } else if let first = activeModels.first {
                    selectedModelPath = first.path
                    loadSelectedModel()
                }
            }
        case "reply":
            if let id = ev["id"] as? Int, let cb = pendingCallbacks.removeValue(forKey: id) {
                cb(ev)
            }
        case "stats":
            inferMS = ev["ema_infer_ms"] as? Double ?? (ev["infer_ms"] as? Double ?? 0)
            totalLatencyMS = ev["total_latency_ms"] as? Double ?? 0
            rtf = ev["rtf"] as? Double ?? 0
            ramMB = ev["rss_mb"] as? Double ?? 0
            let workerInDB = ev["input_db"] as? Double ?? Double(inputLevelDB)
            let newGated = ev["gated"] as? Bool ?? false
            isGated = newGated

            let engTag = activeEngine == .beatrice ? "BTR" : "MLX"
            if lastGatedState != newGated {
                lastGatedState = newGated
                if newGated {
                    appendLog(engTag, activeEngine == .beatrice ? "Idle (0% CPU)" : "Idle (0% GPU)")
                } else {
                    appendLog(engTag, String(format: "Active (in %.0f dB)", workerInDB))
                }
            } else if !newGated {
                let now = Date()
                if now.timeIntervalSince(lastTelemetryLogTime) >= 2.0 {
                    lastTelemetryLogTime = now
                    appendLog(
                        engTag,
                        String(
                            format: "in %.0fdB -> out %.0fdB | %.1fms (%.2fx) | %.0fMB",
                            workerInDB, outputLevelDB, inferMS, rtf, ramMB
                        )
                    )
                }
            }
        case "error":
            let err = ev["error"] as? String ?? "Worker error"
            errorMessage = err
            appendLog("ERR", err)
        default:
            break
        }
    }

    private func applyRVCModelList(_ rawList: [[String: Any]]) {
        rvcModels = rawList.compactMap { m in
            guard let path = m["path"] as? String,
                  let name = m["name"] as? String,
                  m["error"] == nil else { return nil }
            return VoiceModelMeta(
                id: path,
                engine: .rvc,
                name: name,
                path: path,
                indexPath: m["index_path"] as? String,
                sampleRate: m["sample_rate"] as? Int ?? 48000,
                version: m["version"] as? String ?? "v2",
                f0: m["f0"] as? Int ?? 1,
                speakers: []
            )
        }
    }

    private func applyBeatriceModelList(_ rawList: [[String: Any]]) {
        beatriceModels = rawList.compactMap { m in
            guard let path = m["path"] as? String,
                  let name = m["name"] as? String,
                  m["error"] == nil else { return nil }
            let rawSpks = m["speakers"] as? [[String: Any]] ?? []
            let speakers: [BeatriceSpeakerMeta] = rawSpks.compactMap { s in
                guard let sid = s["id"] as? Int,
                      let sname = s["name"] as? String,
                      let slabel = s["label"] as? String else { return nil }
                return BeatriceSpeakerMeta(
                    id: sid,
                    name: sname,
                    label: slabel,
                    averagePitchHz: s["average_pitch_hz"] as? Double ?? 200.0
                )
            }
            return VoiceModelMeta(
                id: path,
                engine: .beatrice,
                name: name,
                path: path,
                indexPath: nil,
                sampleRate: 24000,
                version: m["version"] as? String ?? "2.0.0-rc.0",
                f0: 1,
                speakers: speakers
            )
        }
    }

    private func applyModelInfo(_ info: [String: Any], syncIndexRate: Bool) {
        if let bs = info["block_samples_48k"] as? Int {
            currentBlockSamples = bs
            audioEngine.blockSamples = bs
        }
        if let rss = info["rss_mb"] as? Double {
            ramMB = rss
        }
        indexLoaded = info["index_loaded"] as? Bool ?? false
        indexRamMB = info["index_ram_mb"] as? Double ?? 0
        if syncIndexRate, let ir = info["index_rate"] as? Double, activeEngine == .rvc {
            suppressSettingsSave = true
            indexRate = ir
            suppressSettingsSave = false
        }
    }
	
    
		
	
    // engine / model / param controls

    func switchEngine(to newEngine: VoiceEngineType) {
        guard newEngine != activeEngine else { return }
        saveCurrentModelSettings()
        activeEngine = newEngine
        guard workerReady else { return }

        isLoadingModel = true
        errorMessage = nil
        statusMessage = "Switching..."
        appendLog("INFO", "Engine -> \(newEngine.title)")

        sendCommand(["cmd": "set_engine", "engine": newEngine.rawValue]) { [weak self] reply in
            guard let self else { return }
            self.isLoadingModel = false
            if reply["ok"] as? Bool == true {
                if let info = reply["info"] as? [String: Any] {
                    self.applyModelInfo(info, syncIndexRate: false)
                }
                if let first = self.activeModels.first {
                    self.selectedModelPath = first.path
                    self.loadSelectedModel()
                } else {
                    self.selectedModelPath = ""
                    self.statusMessage = "No models"
                    self.appendLog("WARN", "No \(newEngine.title) models found")
                }
            } else {
                let err = reply["error"] as? String ?? "Engine switch failed"
                self.errorMessage = err
                self.appendLog("ERR", err)
            }
        }
    }

    func loadSelectedModel() {
        guard workerReady, !selectedModelPath.isEmpty else { return }
        UserDefaults.standard.set(selectedModelPath, forKey: "SwiftRVC.lastModelPath")
        let model = selectedModel
        let name = model?.name ?? "model"
        let restored = restoreSettingsIfSaved(for: selectedModelPath, modelName: name)
        
        // pick a sensible default female speaker on first beatrice load
        if activeEngine == .beatrice && !restored, let spks = model?.speakers, !spks.isEmpty {
            let best = spks.first(where: { $0.averagePitchHz >= 220.0 }) ?? spks[0]
            suppressSettingsSave = true
            selectedSpeakerID = best.id
            suppressSettingsSave = false
        }

        isLoadingModel = true
        errorMessage = nil
        statusMessage = "Loading..."
        let t0 = Date()
        let tag = activeEngine == .beatrice ? "BTR" : "MLX"
        appendLog(tag, "Loading \(name)...")

        var cmd: [String: Any] = [
            "cmd": "load_model",
            "engine": activeEngine.rawValue,
            "path": selectedModelPath
        ]
        if activeEngine == .beatrice {
            cmd["speaker_id"] = selectedSpeakerID
        } else if let idx = model?.indexPath {
            cmd["index_path"] = idx
        }

        sendCommand(cmd) { [weak self] reply in
            guard let self else { return }
            self.isLoadingModel = false
            if reply["ok"] as? Bool == true {
                if let info = reply["info"] as? [String: Any] {
                    self.applyModelInfo(info, syncIndexRate: !restored)
                }
                let dt = Date().timeIntervalSince(t0)
                self.pushParamsNow()
                self.statusMessage = self.isStreaming ? "Live" : "Ready"
                if self.activeEngine == .beatrice {
                    let spkName = self.selectedModel?.speakers.first(where: { $0.id == self.selectedSpeakerID })?.label ?? "#\(self.selectedSpeakerID)"
                    self.appendLog("BTR", String(format: "Loaded %@ (%@) in %.2fs (%.0f MB)", name, spkName, dt, self.ramMB))
                } else {
                    let idxStr = self.indexLoaded ? String(format: ", index %.1f MB", self.indexRamMB) : ""
                    self.appendLog("MLX", String(format: "Loaded %@ in %.2fs (%.0f MB%@)", name, dt, self.ramMB, idxStr))
                }
            } else {
                let err = reply["error"] as? String ?? "Failed to load model"
                self.errorMessage = err
                self.statusMessage = "Load failed"
                self.appendLog("ERR", "Load failed: \(err)")
            }
        }
    }

    func changeSpeaker(to speakerID: Int) {
        selectedSpeakerID = speakerID
        saveCurrentModelSettings()
        guard workerReady, activeEngine == .beatrice else { return }
        sendCommand(["cmd": "set_speaker", "speaker_id": speakerID]) { [weak self] reply in
            guard let self else { return }
            if reply["ok"] as? Bool == true {
                let label = self.selectedModel?.speakers.first(where: { $0.id == speakerID })?.label ?? "#\(speakerID)"
                self.appendLog("BTR", "Voice -> \(label)")
            }
        }
    }

    func changePreset(to newPreset: EfficiencyPreset) {
        preset = newPreset
        UserDefaults.standard.set(newPreset.rawValue, forKey: "SwiftRVC.lastPreset")
        guard workerReady else { return }
        isLoadingModel = true
        statusMessage = "Switching..."
        let tag = activeEngine == .beatrice ? "BTR" : "MLX"
        appendLog(tag, "Preset -> \(newPreset.title)")

        sendCommand(["cmd": "set_preset", "preset": newPreset.rawValue]) { [weak self] reply in
            guard let self else { return }
            self.isLoadingModel = false
            if reply["ok"] as? Bool == true {
                if let info = reply["info"] as? [String: Any] {
                    self.applyModelInfo(info, syncIndexRate: true)
                }
                self.pushParamsNow()
                self.statusMessage = self.isStreaming ? "Live" : "Ready"
                let info = reply["info"] as? [String: Any]
                let f0 = info?["f0_method"] as? String ?? ""
                let la = info?["lookahead_ms"] as? Int ?? 0
                self.appendLog(tag, String(format: "%@ (%dms, %dms la, %@, %.0f MB)", newPreset.title, self.currentBlockSamples / 48, la, f0, self.ramMB))
            } else {
                let err = reply["error"] as? String ?? "Preset switch failed"
                self.errorMessage = err
                self.appendLog("ERR", err)
            }
        }
    }

    func scheduleParamPush() {
        audioEngine.bypassDry = bypass
        saveCurrentModelSettings()
        paramDebounceTask?.cancel()
        paramDebounceTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 35_000_000)
            guard !Task.isCancelled else { return }
            self?.pushParamsNow()
        }
    }

    func pushParamsNow() {
        audioEngine.bypassDry = bypass
        guard workerReady else { return }
        let allowIdx = activeEngine == .rvc && preset != .ultraPerformance && preset != .performance
        sendCommand([
            "cmd": "set_params",
            "pitch_shift": pitchShift,
            "formant_shift": activeEngine == .beatrice ? max(-2.0, min(2.0, formantShift)) : formantShift,
            "pitch_floor_hz": pitchFloorHz,
            "gate_db": gateDB,
            "index_rate": allowIdx ? indexRate : 0.0,
            "protect": protect,
            "noise_scale": noiseScale,
            "intonation": intonation,
            "voice_polish": voicePolish,
            "rms_mix": rmsMix,
            "output_gain": outputGain,
            "bypass": bypass
        ])
    }

    // streaming

    func toggleStreaming() {
        if isStreaming { stopStreaming() } else { startStreaming() }
    }

    func startStreaming() {
        guard workerReady, audioSocketReady else {
            pendingStartOutputRecording = false
            errorMessage = "Worker not ready"
            appendLog("WARN", "Worker not ready")
            return
        }
        guard !selectedModelPath.isEmpty else {
            pendingStartOutputRecording = false
            errorMessage = "Select or add a model first"
            appendLog("WARN", "No model selected")
            return
        }
        guard !isExportingWav, !isRecordingSample else {
            pendingStartOutputRecording = false
            return
        }

        switch inputSourceMode {
        case .mic:
            let authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
            switch authStatus {
            case .authorized:
                beginAudioSession()
            case .notDetermined:
                appendLog("AUDIO", "Requesting mic permission...")
                AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                    Task { @MainActor in
                        if granted {
                            self?.beginAudioSession()
                        } else {
                            self?.pendingStartOutputRecording = false
                            let msg = "Mic permission denied. Enable SwiftRVC in System Settings > Privacy & Security > Microphone."
                            self?.errorMessage = msg
                            self?.appendLog("ERR", msg)
                        }
                    }
                }
            default:
                pendingStartOutputRecording = false
                let msg = "Mic permission denied. Enable SwiftRVC in System Settings > Privacy & Security > Microphone."
                errorMessage = msg
                appendLog("ERR", msg)
            }

        case .file:
            guard !loadedFileSamples.isEmpty else {
                pendingStartOutputRecording = false
                errorMessage = "Open an audio file or record a sample first"
                return
            }
            beginAudioSession()

        case .systemApp:
            if tappableApps.isEmpty {
                refreshTappableApps(logCount: false)
            }
            beginAudioSession()
        }
    }

    private func beginAudioSession() {
        guard let outDev = outputDevices.first(where: { $0.id == selectedOutputID }) else {
            pendingStartOutputRecording = false
            errorMessage = "Output device unavailable"
            appendLog("ERR", errorMessage ?? "")
            return
        }
        let inDev = inputDevices.first(where: { $0.id == selectedInputID })
        if inputSourceMode == .mic && inDev == nil {
            pendingStartOutputRecording = false
            errorMessage = "Mic unavailable"
            appendLog("ERR", errorMessage ?? "")
            return
        }
        let tapTarget = tappableApps.first(where: { $0.id == selectedTapAppID }) ?? tappableApps.first
        let monDev = outputDevices.first(where: { $0.id == selectedMonitorID })

        errorMessage = nil
        lastGatedState = nil
        sendCommand(["cmd": "reset"])
        pushParamsNow()

        audioEngine.monitorEnabled = monitorEnabled
        audioEngine.bypassDry = bypass

        do {
            try audioEngine.start(
                sourceMode: inputSourceMode,
                inputDevice: inDev,
                tapTarget: tapTarget,
                muteTappedApp: muteTappedApp,
                fileSamples: loadedFileSamples,
                fileLoop: fileLoop,
                outputDevice: outDev,
                monitorDevice: monDev,
                socketPath: socketPath,
                blockSamples: currentBlockSamples
            )
            isStreaming = true
            if inputSourceMode == .file {
                fileIsPlaying = true
            }
            statusMessage = bypass ? "Bypassed" : "Live"
            if pendingStartOutputRecording {
                pendingStartOutputRecording = false
                beginOutputRecording()
            }
        } catch {
            pendingStartOutputRecording = false
            errorMessage = error.localizedDescription
            isStreaming = false
            statusMessage = "Error"
            appendLog("ERR", "Start failed: \(error.localizedDescription)")
        }
    }

    func stopStreaming() {
        if isRecordingOutput {
            stopOutputRecording()
        }
        audioEngine.stop()
        isStreaming = false
        inputLevelDB = -80.0
        outputLevelDB = -80.0
        isGated = true
        lastGatedState = nil
        if workerReady {
            statusMessage = "Stopped"
        }
    }

    func restartAudioIfRunning() {
        if isStreaming {
            appendLog("AUDIO", "Restarting audio...")
            stopStreaming()
            startStreaming()
        }
    }
	
    
    
		
    
	
    // live output recorder

    func toggleOutputRecording() {
        if isRecordingOutput {
            stopOutputRecording()
            return
        }
        stopPreviewLastRecording()
        errorMessage = nil
        if isStreaming {
            beginOutputRecording()
        } else {
            pendingStartOutputRecording = true
            startStreaming()
        }
    }

    private func beginOutputRecording() {
        let ts = fileTimestampFormatter.string(from: Date())
        let rawModel = selectedModel?.name ?? "voice"
        let cleanModel = rawModel
            .replacingOccurrences(of: " ", with: "_")
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
        
        let spkSuffix: String
        if activeEngine == .beatrice,
           let spk = selectedModel?.speakers.first(where: { $0.id == selectedSpeakerID }) {
            let cleanSpk = spk.name.replacingOccurrences(of: " ", with: "_").replacingOccurrences(of: "/", with: "-")
            spkSuffix = "_\(cleanSpk)"
        } else {
            spkSuffix = ""
        }
        let filename = "\(ts)_\(cleanModel)\(spkSuffix).wav"
        let outURL = recordingsDirURL.appendingPathComponent(filename)

        do {
            try audioEngine.startOutputRecording(to: outURL)
            isRecordingOutput = true
            outputRecordSeconds = 0.0
            statusMessage = bypass ? "Rec (Bypass)" : "Recording"
        } catch {
            isRecordingOutput = false
            let msg = "Record failed: \(error.localizedDescription)"
            errorMessage = msg
            appendLog("ERR", msg)
        }
    }

    func stopOutputRecording() {
        guard isRecordingOutput else { return }
        isRecordingOutput = false
        audioEngine.stopOutputRecording { [weak self] url, dur, err in
            guard let self else { return }
            if let err {
                self.errorMessage = "Record error: \(err.localizedDescription)"
            }
            if let url {
                self.lastRecordedOutputURL = url
                self.lastRecordedOutputDuration = dur
                let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
                let bytes = (attrs?[.size] as? NSNumber)?.doubleValue ?? 0
                let mb = bytes / (1024.0 * 1024.0)
                self.appendLog("AUDIO", String(format: "Saved %@ (%.1fs, %.2f MB)", url.lastPathComponent, dur, mb))
            }
            if self.isStreaming {
                self.statusMessage = self.bypass ? "Bypassed" : "Live"
            } else if self.workerReady {
                self.statusMessage = "Saved"
            }
        }
    }

    func togglePreviewLastRecording() {
        if isPreviewingRecordedOutput {
            stopPreviewLastRecording()
            return
        }
        guard let url = lastRecordedOutputURL, FileManager.default.fileExists(atPath: url.path) else {
            appendLog("WARN", "Take not found")
            return
        }
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            let del = OutputPreviewPlayerDelegate()
            del.onFinish = { [weak self] in
                self?.isPreviewingRecordedOutput = false
                self?.outputPreviewPlayer = nil
            }
            player.delegate = del
            player.prepareToPlay()
            guard player.play() else {
                appendLog("ERR", "Preview playback failed")
                return
            }
            outputPreviewDelegate = del
            outputPreviewPlayer = player
            isPreviewingRecordedOutput = true
        } catch {
            appendLog("ERR", "Preview error: \(error.localizedDescription)")
        }
    }

    func stopPreviewLastRecording() {
        outputPreviewPlayer?.stop()
        outputPreviewPlayer = nil
        isPreviewingRecordedOutput = false
    }

    func revealLastRecordingInFinder() {
        if let url = lastRecordedOutputURL, FileManager.default.fileExists(atPath: url.path) {
            NSWorkspace.shared.activateFileViewerSelecting([url])
        } else {
            try? FileManager.default.createDirectory(at: recordingsDirURL, withIntermediateDirectories: true)
            NSWorkspace.shared.open(recordingsDirURL)
        }
    }

    func saveLastRecordingAsPanel() {
        guard let srcURL = lastRecordedOutputURL, FileManager.default.fileExists(atPath: srcURL.path) else { return }
        let panel = NSSavePanel()
        panel.title = "Save Recording"
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = srcURL.lastPathComponent
        guard panel.runModal() == .OK, let dstURL = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: dstURL.path) {
                try FileManager.default.removeItem(at: dstURL)
            }
            try FileManager.default.copyItem(at: srcURL, to: dstURL)
            appendLog("AUDIO", "Saved copy -> \(dstURL.lastPathComponent)")
        } catch {
            errorMessage = "Save failed: \(error.localizedDescription)"
            appendLog("ERR", errorMessage ?? "")
        }
    }
    
	
		
    
    // file source & offline wav export

    func chooseAudioFilePanel() {
        let panel = NSOpenPanel()
        panel.title = "Open Audio / Video File"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false

        var types: [UTType] = [.audio, .movie, .wav, .mp3, .aiff, .mpeg4Audio, .mpeg4Movie, .quickTimeMovie]
        for ext in ["m4a", "flac", "ogg", "opus", "webm", "caf", "aac", "mkv"] {
            if let t = UTType(filenameExtension: ext) {
                types.append(t)
            }
        }
        panel.allowedContentTypes = types

        guard panel.runModal() == .OK, let url = panel.url else { return }
        loadAudioFile(url: url, displayName: nil)
    }

    func loadAudioFile(url: URL, displayName: String? = nil) {
        let label = displayName ?? url.lastPathComponent
        errorMessage = nil
        appendLog("AUDIO", "Decoding \(label)...")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                let samples = try AudioFileDecoder.decodeTo48kMono(url: url)
                guard !samples.isEmpty else {
                    throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "File has no audio"])
                }
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    self.loadedFileURL = url
                    self.loadedFileName = label
                    self.loadedFileSamples = samples
                    self.fileTotalSamples = samples.count
                    self.filePositionSamples = 0
                    self.audioEngine.updateLoadedFileSamples(samples)
                    let dur = Double(samples.count) / 48000.0
                    self.appendLog("AUDIO", String(format: "Loaded %@ (%.1fs)", label, dur))
                }
            } catch {
                Task { @MainActor [weak self] in
                    let msg = "Decode failed (\(label)): \(error.localizedDescription)"
                    self?.errorMessage = msg
                    self?.appendLog("ERR", msg)
                }
            }
        }
    }

    func toggleFilePlayPause() {
        let next = !fileIsPlaying
        fileIsPlaying = next
        audioEngine.setFilePlayback(playing: next)
    }

    func seekFile(toFraction frac: Double) {
        audioEngine.seekFile(toFraction: frac)
    }

    func syncFileLoop() {
        audioEngine.setFileLoop(fileLoop)
    }

    func toggleRecordMicSample() {
        if isRecordingSample {
            finishMicSampleRecording(autoSwitchToFile: true)
            return
        }

        if isStreaming {
            stopStreaming()
        }
        errorMessage = nil

        let authStatus = AVCaptureDevice.authorizationStatus(for: .audio)
        switch authStatus {
        case .authorized:
            beginMicSampleRecording()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                Task { @MainActor in
                    if granted {
                        self?.beginMicSampleRecording()
                    } else {
                        let msg = "Mic permission denied. Enable SwiftRVC in System Settings > Privacy & Security > Microphone."
                        self?.errorMessage = msg
                        self?.appendLog("ERR", msg)
                    }
                }
            }
        default:
            let msg = "Mic permission denied. Enable SwiftRVC in System Settings > Privacy & Security > Microphone."
            errorMessage = msg
            appendLog("ERR", msg)
        }
    }

    private func beginMicSampleRecording() {
        do {
            if FileManager.default.fileExists(atPath: sampleRecordURL.path) {
                try FileManager.default.removeItem(at: sampleRecordURL)
            }
            let settings: [String: Any] = [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: 48000.0,
                AVNumberOfChannelsKey: 1,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false
            ]
            let rec = try AVAudioRecorder(url: sampleRecordURL, settings: settings)
            guard rec.record() else {
                throw NSError(domain: "SwiftRVC", code: -1, userInfo: [NSLocalizedDescriptionKey: "Recorder failed to start"])
            }
            sampleRecorder = rec
            isRecordingSample = true
            recordingSeconds = 0.0
            statusMessage = "Recording mic..."
            appendLog("AUDIO", "Recording mic sample...")

            recordTimer?.invalidate()
            recordTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
                Task { @MainActor [weak self] in
                    guard let self, let r = self.sampleRecorder, r.isRecording else { return }
                    self.recordingSeconds = r.currentTime
                }
            }
        } catch {
            errorMessage = "Mic record failed: \(error.localizedDescription)"
            appendLog("ERR", errorMessage ?? "")
        }
    }

    private func finishMicSampleRecording(autoSwitchToFile: Bool) {
        recordTimer?.invalidate()
        recordTimer = nil
        let dur = sampleRecorder?.currentTime ?? recordingSeconds
        sampleRecorder?.stop()
        sampleRecorder = nil
        isRecordingSample = false
        statusMessage = workerReady ? "Ready" : statusMessage
        appendLog("AUDIO", String(format: "Mic sample: %.1fs", dur))
        if autoSwitchToFile {
            inputSourceMode = .file
        }
        loadAudioFile(url: sampleRecordURL, displayName: "Mic Sample")
    }

    func exportConvertedWavPanel() {
        guard workerReady, audioSocketReady, !selectedModelPath.isEmpty, !loadedFileSamples.isEmpty, !isExportingWav else {
            return
        }

        let panel = NSSavePanel()
        panel.title = "Export WAV"
        panel.allowedContentTypes = [.wav]
        let baseStem = loadedFileURL?.deletingPathExtension().lastPathComponent
            ?? (loadedFileName.isEmpty ? "voice" : loadedFileName.replacingOccurrences(of: " ", with: "_"))
        let modelTag = selectedModel?.name.replacingOccurrences(of: " ", with: "_") ?? "converted"
        panel.nameFieldStringValue = "\(baseStem)_\(modelTag).wav"

        guard panel.runModal() == .OK, let outURL = panel.url else { return }

        if isStreaming {
            stopStreaming()
        }
        errorMessage = nil
        isExportingWav = true
        exportProgress = 0.0
        statusMessage = "Exporting..."
        
        let samples = loadedFileSamples
        let sock = socketPath
        let bs = currentBlockSamples
        let engine = audioEngine
        let t0 = Date()
        appendLog("AUDIO", "Exporting -> \(outURL.lastPathComponent)")

        pushParamsNow()
        sendCommand(["cmd": "reset"]) { [weak self] _ in
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    // let worker socket finish closing previous connection if any
                    usleep(25_000)
                    let converted = try engine.convertFileOffline(
                        samples: samples,
                        socketPath: sock,
                        blockSamples: bs,
                        onProgress: { [weak self] frac in
                            Task { @MainActor [weak self] in
                                self?.exportProgress = frac
                                self?.statusMessage = String(format: "Exporting %.0f%%", frac * 100.0)
                            }
                        }
                    )
                    try AudioFileDecoder.writeWav48kMono(samples: converted, to: outURL)
                    let dt = Date().timeIntervalSince(t0)
                    let dur = Double(samples.count) / 48000.0
                    let speed = dt > 0 ? dur / dt : 0
                    Task { @MainActor [weak self] in
                        self?.isExportingWav = false
                        self?.exportProgress = 1.0
                        self?.statusMessage = "Exported"
                        self?.appendLog("AUDIO", String(format: "Exported %@ (%.1fs in %.1fs, %.1fx)", outURL.lastPathComponent, dur, dt, speed))
                    }
                } catch {
                    Task { @MainActor [weak self] in
                        self?.isExportingWav = false
                        let msg = "Export failed: \(error.localizedDescription)"
                        self?.errorMessage = msg
                        self?.statusMessage = "Export failed"
                        self?.appendLog("ERR", msg)
                    }
                }
            }
        }
    }

    // model import

    func importModelPanel() {
        let panel = NSOpenPanel()
        panel.title = activeEngine == .beatrice
            ? "Add Beatrice Model (.zip or .toml)"
            : "Add RVC Model (.pth, .zip, or .index)"
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.zip, .data]

        guard panel.runModal() == .OK, let url = panel.url else { return }
        let ext = url.pathExtension.lowercased()
        let stem = url.deletingPathExtension().lastPathComponent

        do {
            if activeEngine == .beatrice {
                let destDir = "\(projectRoot)/beatrice_models/\(stem)"
                try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)
                if ext == "zip" {
                    appendLog("INFO", "Extracting \(url.lastPathComponent)...")
                    let proc = Process()
                    proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
                    proc.arguments = ["-o", url.path, "-d", destDir]
                    try proc.run()
                    proc.waitUntilExit()
                } else if ext == "toml" {
                    let srcDir = url.deletingLastPathComponent()
                    let items = try FileManager.default.contentsOfDirectory(atPath: srcDir.path)
                    for item in items where item.hasSuffix(".toml") || item.hasSuffix(".bin") {
                        let dst = "\(destDir)/\(item)"
                        if FileManager.default.fileExists(atPath: dst) { try FileManager.default.removeItem(atPath: dst) }
                        try FileManager.default.copyItem(atPath: srcDir.appendingPathComponent(item).path, toPath: dst)
                    }
                } else {
                    errorMessage = "Pick a Beatrice .zip or .toml file"
                    return
                }
                rescanAndSelect(prefix: destDir, engine: .beatrice)
                return
            }

            if ext == "index" {
                guard !selectedModelPath.isEmpty else {
                    errorMessage = "Select a .pth model first"
                    return
                }
                let modelDir = URL(fileURLWithPath: selectedModelPath).deletingLastPathComponent().path
                let dst = "\(modelDir)/\(url.lastPathComponent)"
                if FileManager.default.fileExists(atPath: dst) { try FileManager.default.removeItem(atPath: dst) }
                try FileManager.default.copyItem(atPath: url.path, toPath: dst)
                appendLog("INFO", "Attached \(url.lastPathComponent)")
                sendCommand(["cmd": "list_models"]) { [weak self] reply in
                    guard let self else { return }
                    if let list = reply["models"] as? [[String: Any]] { self.applyRVCModelList(list) }
                    self.loadSelectedModel()
                }
                return
            }

            let destDir = "\(projectRoot)/models/\(stem)"
            try FileManager.default.createDirectory(atPath: destDir, withIntermediateDirectories: true)
            if ext == "zip" {
                appendLog("INFO", "Extracting \(url.lastPathComponent)...")
                let proc = Process()
                proc.executableURL = URL(fileURLWithPath: "/usr/bin/unzip")
                proc.arguments = ["-o", url.path, "-d", destDir]
                try proc.run()
                proc.waitUntilExit()
            } else if ext == "pth" {
                let dst = "\(destDir)/\(url.lastPathComponent)"
                if FileManager.default.fileExists(atPath: dst) { try FileManager.default.removeItem(atPath: dst) }
                try FileManager.default.copyItem(atPath: url.path, toPath: dst)
            } else {
                errorMessage = "Pick a .pth, .zip, or .index file"
                return
            }
            rescanAndSelect(prefix: destDir, engine: .rvc)
        } catch {
            errorMessage = "Import failed: \(error.localizedDescription)"
            appendLog("ERR", errorMessage ?? "")
        }
    }

    private func rescanAndSelect(prefix: String, engine: VoiceEngineType) {
        sendCommand(["cmd": "list_models"]) { [weak self] reply in
            guard let self else { return }
            if let list = reply["models"] as? [[String: Any]] { self.applyRVCModelList(list) }
            if let blist = reply["beatrice_models"] as? [[String: Any]] { self.applyBeatriceModelList(blist) }
            let pool = engine == .beatrice ? self.beatriceModels : self.rvcModels
            if let added = pool.first(where: { $0.path.hasPrefix(prefix) }) {
                self.selectedModelPath = added.path
                self.loadSelectedModel()
            }
        }
    }
}
