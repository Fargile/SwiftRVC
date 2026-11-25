import AppKit
import SwiftUI

// note: avoid @State here so command-line-tools builds work without xcode swiftui macros
struct ContentView: View {
    @ObservedObject var ctrl: VoiceChangerController

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            ScrollView(.vertical, showsIndicators: false) {
                controlsColumn
                    .padding(16)
            }
            .frame(width: 420)

            if ctrl.showLogPanel {
                Divider()
                logPanelColumn
                    .frame(minWidth: 280, idealWidth: 320, maxWidth: .infinity)
            }
        }
        .frame(minHeight: 620)
    }
	
    private var controlsColumn: some View {
        VStack(alignment: .leading, spacing: 11) {
            headerSection
            Divider()
            audioRoutingSection
            Divider()
            engineModelAndPresetSection
            Divider()
            tuningSection
            Divider()
            actionSection
            telemetryFooter
        }
    }

    private var headerSection: some View {
        HStack {
            HStack(spacing: 8) {
                Circle()
                    .fill(statusDotColor)
                    .frame(width: 9, height: 9)
                Text("SwiftRVC")
                    .font(.headline)
                Text(ctrl.activeEngine.badge)
                    .font(.caption2.monospaced())
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(ctrl.activeEngine == .beatrice ? Color.purple.opacity(0.22) : Color.secondary.opacity(0.15))
                    .cornerRadius(4)
            }
            Spacer()
            Text(ctrl.statusMessage)
                .font(.caption)
                .foregroundStyle(.secondary)
            Button(ctrl.showLogPanel ? "Hide Log" : "Log") {
                ctrl.showLogPanel.toggle()
            }
            .buttonStyle(.bordered)
            .controlSize(.mini)
        }
    }

    private var statusDotColor: Color {
        if ctrl.errorMessage != nil { return .red }
        if ctrl.isLoadingModel { return .orange }
        if ctrl.isRecordingOutput { return .red }
        if ctrl.isStreaming { return ctrl.bypass ? .yellow : .green }
        return ctrl.workerReady ? .secondary.opacity(0.5) : .orange
    }

    // audio routing

    private var audioRoutingSection: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text("Audio")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button("Test") { ctrl.playTestTone() }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    .disabled(!ctrl.isStreaming)
                    .help("Play a 440Hz tone to output and monitor")
                Button("Refresh") { ctrl.refreshDevices() }
                    .buttonStyle(.link)
                    .font(.caption)
            }

            HStack {
                Text("Source")
                    .frame(width: 56, alignment: .leading)
                    .foregroundStyle(.secondary)
                Picker("Source", selection: Binding(
                    get: { ctrl.inputSourceMode },
                    set: { ctrl.switchInputSourceMode(to: $0) }
                )) {
                    ForEach(InputSourceMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .disabled(ctrl.isExportingWav)
            }
            
            if ctrl.inputSourceMode == .mic {
                HStack {
                    Text("Mic")
                        .frame(width: 56, alignment: .leading)
                        .foregroundStyle(.secondary)
                    Picker("", selection: $ctrl.selectedInputID) {
                        ForEach(ctrl.inputDevices) { dev in
                            Text(dev.displayLabel).tag(dev.id)
                        }
                    }
                    .labelsHidden()
                    .onChange(of: ctrl.selectedInputID) { _ in ctrl.restartAudioIfRunning() }
                }
            } else if ctrl.inputSourceMode == .file {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 6) {
                        Button("Open...") { ctrl.chooseAudioFilePanel() }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .disabled(ctrl.isExportingWav || ctrl.isRecordingSample)

                        Button(ctrl.isRecordingSample
                               ? String(format: "Stop (%.0fs)", ctrl.recordingSeconds)
                               : "Rec Mic") {
                            ctrl.toggleRecordMicSample()
                        }
                        .buttonStyle(.bordered)
                        .tint(ctrl.isRecordingSample ? .red : nil)
                        .controlSize(.small)
                        .disabled(ctrl.isExportingWav)
                        .help("Record a short mic sample to loop while tuning")

                        Spacer()

                        Button(ctrl.isExportingWav
                               ? String(format: "%.0f%%", ctrl.exportProgress * 100.0)
                               : "Export WAV") {
                            ctrl.exportConvertedWavPanel()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.small)
                        .disabled(
                            !ctrl.workerReady || ctrl.isLoadingModel
                            || ctrl.selectedModelPath.isEmpty
                            || ctrl.loadedFileSamples.isEmpty
                            || ctrl.isExportingWav || ctrl.isRecordingSample
                        )
                        .help("Convert the full file offline to 48kHz WAV")
                    }

                    if !ctrl.loadedFileSamples.isEmpty {
                        HStack(spacing: 6) {
                            Button(action: { ctrl.toggleFilePlayPause() }) {
                                Image(systemName: ctrl.fileIsPlaying ? "pause.fill" : "play.fill")
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                            .disabled(!ctrl.isStreaming)

                            Text(ctrl.loadedFileName)
                                .font(.caption.weight(.medium))
                                .lineLimit(1)
                                .truncationMode(.middle)

                            Spacer()

                            Toggle("Loop", isOn: $ctrl.fileLoop)
                                .toggleStyle(.checkbox)
                                .controlSize(.small)
                                .font(.caption)
                                .onChange(of: ctrl.fileLoop) { _ in ctrl.syncFileLoop() }
                        }

                        HStack(spacing: 6) {
                            Slider(value: Binding(
                                get: { ctrl.fileProgressFraction },
                                set: { ctrl.seekFile(toFraction: $0) }
                            ), in: 0...1)
                            .controlSize(.mini)

                            Text("\(ctrl.fileCurrentTimeText)/\(ctrl.fileTotalTimeText)")
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                                .frame(width: 70, alignment: .trailing)
                        }
                    } else {
                        Text("Open an audio/video file or record a quick mic sample.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .padding(7)
                .background(Color.secondary.opacity(0.08))
                .cornerRadius(6)
            } else {
                VStack(alignment: .leading, spacing: 5) {
                    HStack {
                        Text("App")
                            .frame(width: 56, alignment: .leading)
                            .foregroundStyle(.secondary)
                        Picker("", selection: $ctrl.selectedTapAppID) {
                            ForEach(ctrl.tappableApps) { app in
                                Text(app.displayLabel).tag(app.id)
                            }
                        }
                        .labelsHidden()
                        .onChange(of: ctrl.selectedTapAppID) { _ in ctrl.restartAudioIfRunning() }
                        .help("Requires macOS System Audio Recording permission")
                    }

                    HStack {
                        Toggle("Mute original", isOn: $ctrl.muteTappedApp)
                            .toggleStyle(.checkbox)
                            .controlSize(.small)
                            .font(.caption)
                            .onChange(of: ctrl.muteTappedApp) { _ in ctrl.restartAudioIfRunning() }
                            .help("Mute tapped app output so you only hear the converted monitor")
                        Spacer()
                    }
                }
            }

            HStack {
                Text("Output")
                    .frame(width: 56, alignment: .leading)
                    .foregroundStyle(.secondary)
                Picker("", selection: $ctrl.selectedOutputID) {
                    ForEach(ctrl.outputDevices) { dev in
                        Text(dev.displayLabel).tag(dev.id)
                    }
                }
                .labelsHidden()
                .onChange(of: ctrl.selectedOutputID) { _ in ctrl.restartAudioIfRunning() }
            }

            HStack {
                Toggle("Monitor", isOn: $ctrl.monitorEnabled)
                    .toggleStyle(.checkbox)
                    .frame(width: 76, alignment: .leading)
                Picker("", selection: $ctrl.selectedMonitorID) {
                    ForEach(ctrl.outputDevices) { dev in
                        Text(dev.displayLabel).tag(dev.id)
                    }
                }
                .labelsHidden()
                .disabled(!ctrl.monitorEnabled)
                .onChange(of: ctrl.selectedMonitorID) { _ in ctrl.syncMonitorOutput() }
            }

            if !ctrl.hasVirtualCable {
                HStack(spacing: 6) {
                    Image(systemName: "info.circle")
                        .foregroundStyle(.orange)
                    Text("No BlackHole 2ch detected")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Copy brew cmd") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString("brew install blackhole-2ch", forType: .string)
                    }
                    .font(.caption)
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
                .padding(.top, 2)
            }
        }
    }

    // engine, model, preset

    private var engineModelAndPresetSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Engine")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Picker("Engine", selection: Binding(
                    get: { ctrl.activeEngine },
                    set: { ctrl.switchEngine(to: $0) }
                )) {
                    ForEach(VoiceEngineType.allCases) { eng in
                        Text(eng.title).tag(eng)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .frame(width: 210)
                .disabled(!ctrl.workerReady || ctrl.isLoadingModel)
            }

            HStack {
                Text(ctrl.activeEngine == .beatrice ? "Pack" : "Model")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if let m = ctrl.selectedModel {
                    Text(m.badgeText)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            HStack(spacing: 6) {
                Picker("", selection: $ctrl.selectedModelPath) {
                    if ctrl.activeModels.isEmpty {
                        Text(ctrl.activeEngine == .beatrice ? "No Beatrice packs" : "No .pth models").tag("")
                    }
                    ForEach(ctrl.activeModels) { m in
                        Text(m.name).tag(m.path)
                    }
                }
                .labelsHidden()
                .disabled(!ctrl.workerReady || ctrl.isLoadingModel)
                .onChange(of: ctrl.selectedModelPath) { _ in ctrl.loadSelectedModel() }

                Button("+ Add") { ctrl.importModelPanel() }
                    .disabled(!ctrl.workerReady || ctrl.isLoadingModel)
                    .help(ctrl.activeEngine == .beatrice ? "Add Beatrice .zip or .toml" : "Add .pth, .zip, or attach .index")
            }

            if ctrl.activeEngine == .beatrice, let spks = ctrl.selectedModel?.speakers, spks.count > 1 {
                HStack {
                    Text("Voice")
                        .frame(width: 56, alignment: .leading)
                        .foregroundStyle(.secondary)
                    Picker("", selection: Binding(
                        get: { ctrl.selectedSpeakerID },
                        set: { ctrl.changeSpeaker(to: $0) }
                    )) {
                        ForEach(spks) { s in
                            Text(s.label).tag(s.id)
                        }
                    }
                    .labelsHidden()
                    .disabled(!ctrl.workerReady || ctrl.isLoadingModel)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Picker("Preset", selection: Binding(
                    get: { ctrl.preset },
                    set: { ctrl.changePreset(to: $0) }
                )) {
                    ForEach(EfficiencyPreset.allCases) { p in
                        Text(p.shortTitle).tag(p)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .controlSize(.small)
                .disabled(!ctrl.workerReady || ctrl.isLoadingModel)

                Text(ctrl.preset.summary(for: ctrl.activeEngine))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 2)
        }
    }
	
    
		
    
    // voice controls

    private var tuningSection: some View {
        VStack(alignment: .leading, spacing: 9) {
            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("Pitch")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    HStack(spacing: 3) {
                        ForEach([-12.0, 0.0, 6.0, 12.0, 18.0], id: \.self) { val in
                            Button(val > 0 ? "+\(Int(val))" : "\(Int(val))") {
                                ctrl.pitchShift = val
                                ctrl.scheduleParamPush()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                    }
                    Text(String(format: "%+.0f st", ctrl.pitchShift))
                        .font(.caption.monospacedDigit())
                        .frame(width: 44, alignment: .trailing)
                }
                Slider(value: $ctrl.pitchShift, in: -24...24, step: 1)
                    .onChange(of: ctrl.pitchShift) { _ in ctrl.scheduleParamPush() }
            }

            VStack(alignment: .leading, spacing: 3) {
                let fRange: ClosedRange<Double> = ctrl.activeEngine == .beatrice ? -2.0...2.0 : -3.0...3.0
                let fStep: Double = ctrl.activeEngine == .beatrice ? 0.5 : 0.25
                HStack {
                    Text("Formant")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    if abs(ctrl.formantShift) > 0.01 {
                        Button("Reset") {
                            ctrl.formantShift = 0.0
                            ctrl.scheduleParamPush()
                        }
                        .buttonStyle(.bordered)
                        .controlSize(.mini)
                    }
                    Text(String(format: "%+.2f st", ctrl.formantShift))
                        .font(.caption.monospacedDigit())
                        .frame(width: 54, alignment: .trailing)
                }
                Slider(value: Binding(
                    get: { max(fRange.lowerBound, min(fRange.upperBound, ctrl.formantShift)) },
                    set: { ctrl.formantShift = $0; ctrl.scheduleParamPush() }
                ), in: fRange, step: fStep)
            }
            .help("Shifts vocal tract timbre without changing pitch")

            VStack(alignment: .leading, spacing: 3) {
                HStack {
                    Text("Pitch Floor")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    HStack(spacing: 3) {
                        ForEach([(0.0, "Off"), (175.0, "175"), (200.0, "200")], id: \.0) { val, lbl in
                            Button(lbl) {
                                ctrl.pitchFloorHz = val
                                ctrl.scheduleParamPush()
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.mini)
                        }
                    }
                    Text(ctrl.pitchFloorHz < 50 ? "Off" : "\(Int(ctrl.pitchFloorHz))Hz")
                        .font(.caption.monospacedDigit())
                        .frame(width: 44, alignment: .trailing)
                }
                Slider(value: $ctrl.pitchFloorHz, in: 0...250, step: 5)
                    .onChange(of: ctrl.pitchFloorHz) { _ in ctrl.scheduleParamPush() }
            }
            .help("Soft lower F0 limit to reduce vocal fry on deep voices")

            HStack {
                Toggle("Polish", isOn: $ctrl.voicePolish)
                    .toggleStyle(.checkbox)
                    .font(.subheadline.weight(.medium))
                    .onChange(of: ctrl.voicePolish) { _ in
                        ctrl.scheduleParamPush()
                        ctrl.appendLog("AUDIO", ctrl.voicePolish ? "Polish on" : "Polish off")
                    }
                    .help("Input leveler, de-bass, F0 smoothing, and output EQ")
                Spacer()
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Gate")
                        .font(.subheadline.weight(.medium))
                    if ctrl.isStreaming && ctrl.isGated && !ctrl.bypass {
                        Text(ctrl.activeEngine == .beatrice ? "IDLE · 0% CPU" : "IDLE · 0% GPU")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.green)
                    } else if ctrl.isStreaming && !ctrl.bypass {
                        Text("ACTIVE")
                            .font(.caption2.monospaced())
                            .foregroundStyle(.blue)
                    }
                    Spacer()
                    Text("\(Int(ctrl.gateDB)) dB")
                        .font(.caption.monospacedDigit())
                        .frame(width: 44, alignment: .trailing)
                }
                Slider(value: $ctrl.gateDB, in: -70...(-15), step: 1)
                    .onChange(of: ctrl.gateDB) { _ in ctrl.scheduleParamPush() }
                levelMeterBar(levelDB: ctrl.inputLevelDB, gateDB: Float(ctrl.gateDB), label: "In")
                levelMeterBar(levelDB: ctrl.outputLevelDB, gateDB: nil, label: "Out")
            }

            if ctrl.activeEngine == .rvc && ctrl.selectedModel?.indexPath != nil
                && ctrl.preset != .performance && ctrl.preset != .ultraPerformance {
                VStack(alignment: .leading, spacing: 3) {
                    HStack {
                        Text("Index")
                            .font(.subheadline.weight(.medium))
                        Text(String(format: "%.1f MB", ctrl.indexRamMB))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text("\(Int((ctrl.indexRate * 100).rounded()))%")
                            .font(.caption.monospacedDigit())
                            .frame(width: 44, alignment: .trailing)
                    }
                    Slider(value: $ctrl.indexRate, in: 0...0.85, step: 0.05)
                        .onChange(of: ctrl.indexRate) { _ in ctrl.scheduleParamPush() }
                }
            }

            DisclosureGroup("Advanced", isExpanded: $ctrl.showMoreControls) {
                VStack(alignment: .leading, spacing: 7) {
                    if ctrl.activeEngine == .rvc {
                        HStack {
                            Text("Protect")
                                .font(.caption)
                                .frame(width: 84, alignment: .leading)
                            Slider(value: $ctrl.protect, in: 0...0.5, step: 0.02)
                                .onChange(of: ctrl.protect) { _ in ctrl.scheduleParamPush() }
                            Text(String(format: "%.2f", ctrl.protect))
                                .font(.caption.monospacedDigit())
                                .frame(width: 36, alignment: .trailing)
                        }
                        .help("Protects voiceless consonants and breaths")

                        HStack {
                            Text("Shimmer")
                                .font(.caption)
                                .frame(width: 84, alignment: .leading)
                            Slider(value: $ctrl.noiseScale, in: 0.15...0.75, step: 0.05)
                                .onChange(of: ctrl.noiseScale) { _ in ctrl.scheduleParamPush() }
                            Text(String(format: "%.2f", ctrl.noiseScale))
                                .font(.caption.monospacedDigit())
                                .frame(width: 36, alignment: .trailing)
                        }
                        .help("VITS stochastic noise scale")
                    } else {
                        HStack {
                            Text("Intonation")
                                .font(.caption)
                                .frame(width: 84, alignment: .leading)
                            Slider(value: $ctrl.intonation, in: 0.2...1.8, step: 0.05)
                                .onChange(of: ctrl.intonation) { _ in ctrl.scheduleParamPush() }
                            Text(String(format: "%.2fx", ctrl.intonation))
                                .font(.caption.monospacedDigit())
                                .frame(width: 36, alignment: .trailing)
                        }
                    }

                    HStack {
                        Text("RMS Mix")
                            .font(.caption)
                            .frame(width: 84, alignment: .leading)
                        Slider(value: $ctrl.rmsMix, in: 0...1, step: 0.05)
                            .onChange(of: ctrl.rmsMix) { _ in ctrl.scheduleParamPush() }
                        Text("\(Int((ctrl.rmsMix * 100).rounded()))%")
                            .font(.caption.monospacedDigit())
                            .frame(width: 36, alignment: .trailing)
                    }

                    HStack {
                        Text("Gain")
                            .font(.caption)
                            .frame(width: 84, alignment: .leading)
                        Slider(value: $ctrl.outputGain, in: 0.4...2.0, step: 0.05)
                            .onChange(of: ctrl.outputGain) { _ in ctrl.scheduleParamPush() }
                        Text(String(format: "%.2fx", ctrl.outputGain))
                            .font(.caption.monospacedDigit())
                            .frame(width: 36, alignment: .trailing)
                    }
                }
                .padding(.top, 4)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private func levelMeterBar(levelDB: Float, gateDB: Float?, label: String) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.caption2.monospaced())
                .foregroundStyle(.secondary)
                .frame(width: 22, alignment: .leading)
            GeometryReader { geo in
                let normLevel = CGFloat(max(0, min(1, (levelDB + 75) / 75)))
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.secondary.opacity(0.18))
                    Capsule()
                        .fill(meterColor(levelDB: levelDB, gateDB: gateDB))
                        .frame(width: geo.size.width * normLevel)
                    if let g = gateDB {
                        let normGate = CGFloat(max(0, min(1, (g + 75) / 75)))
                        Rectangle()
                            .fill(Color.primary.opacity(0.75))
                            .frame(width: 2)
                            .offset(x: geo.size.width * normGate)
                    }
                }
            }
            .frame(height: 6)
            Text(levelDB <= -74 ? "-inf" : String(format: "%3.0f", levelDB))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 28, alignment: .trailing)
        }
    }

    private func meterColor(levelDB: Float, gateDB: Float?) -> Color {
        if let g = gateDB {
            return levelDB >= g ? .green : .secondary.opacity(0.5)
        }
        return levelDB > -60 ? .blue : .secondary.opacity(0.4)
    }

    // actions & footer

    private var actionSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let err = ctrl.errorMessage {
                Text(err)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack(spacing: 8) {
                Button(action: { ctrl.toggleStreaming() }) {
                    HStack {
                        Image(systemName: startButtonIcon)
                        Text(startButtonLabel)
                            .fontWeight(.semibold)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .tint(ctrl.isStreaming ? .red : .accentColor)
                .disabled(
                    !ctrl.workerReady || ctrl.isLoadingModel || ctrl.selectedModelPath.isEmpty
                    || ctrl.isExportingWav || ctrl.isRecordingSample
                    || (ctrl.inputSourceMode == .file && ctrl.loadedFileSamples.isEmpty)
                )

                Button(action: { ctrl.toggleOutputRecording() }) {
                    HStack(spacing: 4) {
                        Image(systemName: ctrl.isRecordingOutput ? "stop.circle.fill" : "record.circle")
                            .foregroundStyle(ctrl.isRecordingOutput ? .white : .red)
                        Text(ctrl.isRecordingOutput
                             ? VoiceChangerController.formatDuration(ctrl.outputRecordSeconds)
                             : "Rec")
                            .monospacedDigit()
                            .fontWeight(ctrl.isRecordingOutput ? .semibold : .regular)
                    }
                    .padding(.vertical, 4)
                }
                .buttonStyle(.borderedProminent)
                .tint(ctrl.isRecordingOutput ? .red : Color.secondary.opacity(0.28))
                .disabled(
                    !ctrl.workerReady || ctrl.isLoadingModel || ctrl.selectedModelPath.isEmpty
                    || ctrl.isExportingWav || ctrl.isRecordingSample
                    || (!ctrl.isStreaming && ctrl.inputSourceMode == .file && ctrl.loadedFileSamples.isEmpty)
                )
                .help("Record converted voice output to recordings/ as 48kHz WAV")

                Toggle("Bypass", isOn: $ctrl.bypass)
                    .toggleStyle(.button)
                    .onChange(of: ctrl.bypass) { _ in
                        ctrl.scheduleParamPush()
                        ctrl.appendLog("AUDIO", ctrl.bypass ? "Bypass on" : "Bypass off")
                        if ctrl.isStreaming {
                            ctrl.statusMessage = ctrl.isRecordingOutput
                                ? (ctrl.bypass ? "Rec (Bypass)" : "Recording")
                                : (ctrl.bypass ? "Bypassed" : "Live")
                        }
                    }
            }

            if ctrl.isRecordingOutput {
                HStack(spacing: 6) {
                    Circle()
                        .fill(Color.red)
                        .frame(width: 7, height: 7)
                    Text("Recording WAV to recordings/...")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            } else if let takeURL = ctrl.lastRecordedOutputURL {
                HStack(spacing: 6) {
                    Image(systemName: "waveform")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(String(format: "%@ (%.1fs)", takeURL.lastPathComponent, ctrl.lastRecordedOutputDuration))
                        .font(.caption2.monospacedDigit())
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button(ctrl.isPreviewingRecordedOutput ? "Stop" : "Play") {
                        ctrl.togglePreviewLastRecording()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    Button("Save") {
                        ctrl.saveLastRecordingAsPanel()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                    Button("Finder") {
                        ctrl.revealLastRecordingInFinder()
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(Color.secondary.opacity(0.08))
                .cornerRadius(6)
            }
        }
    }

    private var startButtonIcon: String {
        if ctrl.isStreaming { return "stop.fill" }
        switch ctrl.inputSourceMode {
        case .mic: return "mic.fill"
        case .file: return "play.fill"
        case .systemApp: return "waveform"
        }
    }

    private var startButtonLabel: String {
        if ctrl.isStreaming { return "Stop" }
        return ctrl.inputSourceMode == .file ? "Play" : "Start"
    }

    private var telemetryFooter: some View {
        HStack {
            Text(String(format: "%.1fms", ctrl.inferMS))
            Text("·")
            Text(String(format: "~%.0fms", ctrl.totalLatencyMS))
            Text("·")
            Text(String(format: "%.2fx", ctrl.rtf))
                .foregroundStyle(ctrl.rtf > 0.85 ? .orange : .secondary)
            Spacer()
            Text(String(format: "%.0f MB", ctrl.ramMB))
        }
        .font(.caption2.monospacedDigit())
        .foregroundStyle(.secondary)
        .padding(.top, 2)
    }
	
    
        
    
	
    
    // right log panel

    private var logPanelColumn: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Log")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Button("Copy") { ctrl.copyLogs() }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                Button("Clear") { ctrl.clearLogs() }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)

            Divider()

            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(ctrl.logs) { entry in
                            HStack(alignment: .top, spacing: 5) {
                                Text(entry.timestamp)
                                    .foregroundStyle(.secondary)
                                Text(entry.tag)
                                    .foregroundStyle(logTagColor(entry.tag))
                                    .frame(width: 38, alignment: .leading)
                                Text(entry.message)
                                    .foregroundStyle(.primary)
                                    .textSelection(.enabled)
                                    .fixedSize(horizontal: false, vertical: true)
                            }
                            .font(.system(size: 10.5, design: .monospaced))
                            .id(entry.id)
                        }
                    }
                    .padding(10)
                }
                .background(Color(nsColor: .textBackgroundColor).opacity(0.45))
                .onChange(of: ctrl.logs.last?.id) { lastID in
                    if let id = lastID {
                        proxy.scrollTo(id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private func logTagColor(_ tag: String) -> Color {
        switch tag {
        case "ERR": return .red
        case "WARN": return .orange
        case "AUDIO": return .green
        case "MLX": return .blue
        case "BTR": return .purple
        default: return .secondary
        }
    }
}
