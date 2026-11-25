import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var controller: VoiceChangerController!
    private var window: NSWindow?
    private var statusItem: NSStatusItem?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        controller = VoiceChangerController()
        
        let contentView = ContentView(ctrl: controller)
        let hosting = NSHostingView(rootView: contentView)

        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 750, height: 710),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        win.minSize = NSSize(width: 420, height: 620)
        win.title = "SwiftRVC"
        win.isReleasedWhenClosed = false
        win.contentView = hosting
        win.center()
        win.makeKeyAndOrderFront(nil)
        self.window = win
		
        setupMenuBarItem()
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !flag {
            window?.makeKeyAndOrderFront(nil)
        }
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        controller.stopWorker()
    }
	
    
		
    
    private func setupMenuBarItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let btn = item.button {
            btn.image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "SwiftRVC")
        }
        
        let menu = NSMenu()
        menu.addItem(NSMenuItem(title: "Show SwiftRVC", action: #selector(showWindow), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Start / Stop", action: #selector(toggleStartStop), keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "Record / Stop", action: #selector(toggleOutputRecord), keyEquivalent: ""))
        menu.addItem(NSMenuItem(title: "Bypass", action: #selector(toggleBypass), keyEquivalent: ""))
        menu.addItem(NSMenuItem.separator())
        menu.addItem(NSMenuItem(title: "Quit SwiftRVC", action: #selector(quitApp), keyEquivalent: "q"))
        item.menu = menu
        self.statusItem = item
    }

    @objc private func showWindow() {
        window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc private func toggleStartStop() {
        Task { @MainActor in controller.toggleStreaming() }
    }

    @objc private func toggleOutputRecord() {
        Task { @MainActor in controller.toggleOutputRecording() }
    }

    @objc private func toggleBypass() {
        Task { @MainActor in
            controller.bypass.toggle()
            controller.scheduleParamPush()
        }
    }

    @objc private func quitApp() {
        controller.stopWorker()
        NSApp.terminate(nil)
    }
}

@main
enum SwiftRVCEntry {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }
}
