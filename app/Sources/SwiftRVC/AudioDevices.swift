import CoreAudio
import Foundation

struct AudioDeviceInfo: Identifiable, Hashable {
    let id: AudioDeviceID
    let uid: String
    let name: String
    let inputChannels: Int
    let outputChannels: Int
    let sampleRate: Double
    
    var isVirtualCable: Bool {
        let lower = name.lowercased()
        return lower.contains("blackhole")
            || lower.contains("vb-cable")
            || lower.contains("loopback")
            || lower.contains("soundflower")
    }

    var displayLabel: String {
        let srK = Int((sampleRate / 1000.0).rounded())
        return "\(name) (\(srK)kHz)"
    }
}
	
    
	
    
enum AudioDeviceDiscovery {
    static func allDevices() -> [AudioDeviceInfo] {
        var propAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        
        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propAddress,
            0, nil,
            &dataSize
        ) == noErr, dataSize > 0 else {
            return []
        }

        let count = Int(dataSize) / MemoryLayout<AudioDeviceID>.size
        var deviceIDs = [AudioDeviceID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propAddress,
            0, nil,
            &dataSize,
            &deviceIDs
        ) == noErr else {
            return []
        }
		
        var result: [AudioDeviceInfo] = []
        for id in deviceIDs {
            let inCh = channelCount(deviceID: id, scope: kAudioDevicePropertyScopeInput)
            let outCh = channelCount(deviceID: id, scope: kAudioDevicePropertyScopeOutput)
            if inCh == 0 && outCh == 0 { continue }

            let name = stringProperty(deviceID: id, selector: kAudioObjectPropertyName) ?? "Device \(id)"
            let uid = stringProperty(deviceID: id, selector: kAudioDevicePropertyDeviceUID) ?? "\(id)"
            let sr = nominalSampleRate(deviceID: id)
            result.append(AudioDeviceInfo(
                id: id,
                uid: uid,
                name: name,
                inputChannels: inCh,
                outputChannels: outCh,
                sampleRate: sr
            ))
        }
        return result
    }

    static func defaultDeviceID(isInput: Bool) -> AudioDeviceID? {
        var addr = AudioObjectPropertyAddress(
            mSelector: isInput ? kAudioHardwarePropertyDefaultInputDevice : kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &addr,
            0, nil,
            &size,
            &deviceID
        ) == noErr, deviceID != 0 else {
            return nil
        }
        return deviceID
    }
	
    private static func stringProperty(deviceID: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var cfStr: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        
        let status = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &cfStr)
        guard status == noErr, let unmanaged = cfStr else { return nil }
        return unmanaged.takeRetainedValue() as String
    }

    private static func nominalSampleRate(deviceID: AudioDeviceID) -> Double {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var sr: Float64 = 48000.0
        var size = UInt32(MemoryLayout<Float64>.size)
        _ = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &sr)
        return sr
    }

    private static func channelCount(deviceID: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        var dataSize: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &addr, 0, nil, &dataSize) == noErr,
              dataSize > 0 else {
            return 0
        }
        
        let rawPtr = UnsafeMutableRawPointer.allocate(
            byteCount: Int(dataSize),
            alignment: MemoryLayout<AudioBufferList>.alignment
        )
        defer { rawPtr.deallocate() }

        guard AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &dataSize, rawPtr) == noErr else {
            return 0
        }
        let abl = UnsafeMutableAudioBufferListPointer(rawPtr.assumingMemoryBound(to: AudioBufferList.self))
        return abl.reduce(0) { $0 + Int($1.mNumberChannels) }
    }
}
