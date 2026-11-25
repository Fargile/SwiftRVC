import Foundation
import os

// simple fixed-size float ring buffer for coreaudio threads
final class FloatRingBuffer {
    private let capacity: Int
    private let buffer: UnsafeMutablePointer<Float>
    private var readIndex: Int = 0
    private var writeIndex: Int = 0
    private var count: Int = 0
    private var lock = os_unfair_lock_s()
    
    init(capacity: Int) {
        self.capacity = max(capacity, 1024)
        self.buffer = UnsafeMutablePointer<Float>.allocate(capacity: self.capacity)
        self.buffer.initialize(repeating: 0, count: self.capacity)
    }

    deinit {
        buffer.deallocate()
    }
	
    var availableToRead: Int {
        os_unfair_lock_lock(&lock)
        let c = count
        os_unfair_lock_unlock(&lock)
        return c
    }

    func reset() {
        os_unfair_lock_lock(&lock)
        readIndex = 0
        writeIndex = 0
        count = 0
        os_unfair_lock_unlock(&lock)
    }

    // drops oldest samples on overflow so latency doesnt creep up
    @discardableResult
    func write(_ src: UnsafePointer<Float>, count n: Int, maxBacklog: Int? = nil) -> Int {
        guard n > 0 else { return 0 }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        
        let limit = min(maxBacklog ?? capacity, capacity)
        var toWrite = n
        var srcPtr = src
        if toWrite > limit {
            srcPtr = src.advanced(by: toWrite - limit)
            toWrite = limit
        }
		
        let overflow = (count + toWrite) - limit
        if overflow > 0 {
            readIndex = (readIndex + overflow) % capacity
            count -= overflow
        }

        let firstPart = min(toWrite, capacity - writeIndex)
        buffer.advanced(by: writeIndex).update(from: srcPtr, count: firstPart)
        let secondPart = toWrite - firstPart
        if secondPart > 0 {
            buffer.update(from: srcPtr.advanced(by: firstPart), count: secondPart)
        }
        
        writeIndex = (writeIndex + toWrite) % capacity
        count += toWrite
        return count
    }
    
	
        
    
    @discardableResult
    func read(into dst: UnsafeMutablePointer<Float>, count n: Int) -> Int {
        guard n > 0 else { return 0 }
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        
        let toRead = min(n, count)
        if toRead == 0 { return 0 }

        let firstPart = min(toRead, capacity - readIndex)
        dst.update(from: buffer.advanced(by: readIndex), count: firstPart)
        let secondPart = toRead - firstPart
        if secondPart > 0 {
            dst.advanced(by: firstPart).update(from: buffer, count: secondPart)
        }
        
        readIndex = (readIndex + toRead) % capacity
        count -= toRead
        return toRead
    }
	
    // only pops if we have a full block ready
    func readExact(into dst: UnsafeMutablePointer<Float>, count n: Int) -> Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        guard count >= n else { return false }

        let firstPart = min(n, capacity - readIndex)
        dst.update(from: buffer.advanced(by: readIndex), count: firstPart)
        let secondPart = n - firstPart
        if secondPart > 0 {
            dst.advanced(by: firstPart).update(from: buffer, count: secondPart)
        }
        readIndex = (readIndex + n) % capacity
        count -= n
        return true
    }
}
