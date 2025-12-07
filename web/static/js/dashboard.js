/**
 * Fall Detection System Dashboard
 * Real-time WebSocket client for monitoring and alerts
 *
 * Arm Edge AI Healthcare Initiative
 * NXP i.MX93 FRDM + Ethos-U65 NPU
 */

(function() {
    'use strict';

    // ==========================================================================
    // Configuration
    // ==========================================================================

    const CONFIG = {
        reconnectDelay: 3000,
        statsRefreshInterval: 5000,
        alertSoundEnabled: true,
        maxAlertHistory: 50
    };

    // ==========================================================================
    // State
    // ==========================================================================

    let socket = null;
    let connected = false;
    let alertHistory = [];
    let statsInterval = null;

    // ==========================================================================
    // DOM Elements
    // ==========================================================================

    const elements = {
        // Connection
        connectionStatus: document.getElementById('connectionStatus'),
        npuBadge: document.getElementById('npuBadge'),

        // Video
        videoContainer: document.getElementById('videoContainer'),
        videoFeed: document.getElementById('videoFeed'),
        fallOverlay: document.getElementById('fallOverlay'),
        fallConfidence: document.getElementById('fallConfidence'),
        fpsDisplay: document.getElementById('fpsDisplay'),
        inferenceDisplay: document.getElementById('inferenceDisplay'),

        // Status
        systemState: document.getElementById('systemState'),
        activityState: document.getElementById('activityState'),
        bodyAngle: document.getElementById('bodyAngle'),
        keypointCount: document.getElementById('keypointCount'),
        uptimeDisplay: document.getElementById('uptimeDisplay'),
        personIndicator: document.getElementById('personIndicator'),

        // Actions
        testAlertBtn: document.getElementById('testAlertBtn'),
        resetBtn: document.getElementById('resetBtn'),
        fullscreenBtn: document.getElementById('fullscreenBtn'),
        refreshStatsBtn: document.getElementById('refreshStatsBtn'),

        // Alerts
        alertCount: document.getElementById('alertCount'),
        alertList: document.getElementById('alertList'),
        alertSound: document.getElementById('alertSound'),

        // Stats
        totalFrames: document.getElementById('totalFrames'),
        avgFps: document.getElementById('avgFps'),
        avgInference: document.getElementById('avgInference'),
        totalFalls: document.getElementById('totalFalls')
    };

    // ==========================================================================
    // WebSocket Connection
    // ==========================================================================

    function initializeSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        socket = io({
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: CONFIG.reconnectDelay
        });

        socket.on('connect', handleConnect);
        socket.on('disconnect', handleDisconnect);
        socket.on('connected', handleServerMessage);
        socket.on('status_update', handleStatusUpdate);
        socket.on('alert', handleAlert);
        socket.on('error', handleError);
    }

    function handleConnect() {
        console.log('[WebSocket] Connected');
        connected = true;
        updateConnectionStatus('connected', 'Connected');
        socket.emit('request_status');
    }

    function handleDisconnect() {
        console.log('[WebSocket] Disconnected');
        connected = false;
        updateConnectionStatus('disconnected', 'Disconnected');
    }

    function handleServerMessage(data) {
        console.log('[Server]', data.message);
    }

    function handleError(error) {
        console.error('[WebSocket] Error:', error);
        updateConnectionStatus('error', 'Error');
    }

    // ==========================================================================
    // Status Updates
    // ==========================================================================

    function handleStatusUpdate(data) {
        // System state
        updateSystemState(data.state);

        // Activity
        elements.activityState.textContent = formatActivityState(data.activity);

        // Metrics
        if (data.metrics) {
            elements.fpsDisplay.textContent = `${data.metrics.fps} FPS`;
            elements.inferenceDisplay.textContent = `${data.metrics.inference_ms} ms`;
            elements.uptimeDisplay.textContent = formatUptime(data.metrics.uptime);

            // NPU status
            updateNpuStatus(data.metrics.npu_active);
        }

        // Detection
        if (data.detection) {
            elements.bodyAngle.textContent = `${data.detection.body_angle.toFixed(1)}°`;
            elements.keypointCount.textContent = data.detection.keypoint_count;

            // Person indicator
            if (data.detection.person_detected) {
                elements.personIndicator.classList.add('active');
            } else {
                elements.personIndicator.classList.remove('active');
            }

            // Fall overlay
            if (data.detection.is_fall) {
                showFallOverlay(data.detection.fall_confidence);
            } else {
                hideFallOverlay();
            }
        }
    }

    function updateSystemState(state) {
        const stateElement = elements.systemState;
        stateElement.textContent = state.toUpperCase().replace('_', ' ');
        stateElement.className = 'system-state ' + state;
    }

    function updateConnectionStatus(status, text) {
        elements.connectionStatus.className = 'status-indicator ' + status;
        elements.connectionStatus.querySelector('.status-text').textContent = text;
    }

    function updateNpuStatus(active) {
        const badge = elements.npuBadge;
        const statusSpan = badge.querySelector('.npu-status');

        if (active) {
            badge.classList.remove('inactive');
            statusSpan.textContent = 'ACTIVE';
        } else {
            badge.classList.add('inactive');
            statusSpan.textContent = 'OFF';
        }
    }

    // ==========================================================================
    // Fall Overlay
    // ==========================================================================

    function showFallOverlay(confidence) {
        elements.fallOverlay.classList.add('active');
        elements.fallConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(0)}%`;

        // Play alert sound
        if (CONFIG.alertSoundEnabled && elements.alertSound) {
            elements.alertSound.play().catch(() => {});
        }
    }

    function hideFallOverlay() {
        elements.fallOverlay.classList.remove('active');
    }

    // ==========================================================================
    // Alerts
    // ==========================================================================

    function handleAlert(data) {
        console.log('[Alert]', data);

        // Add to history
        alertHistory.unshift(data);
        if (alertHistory.length > CONFIG.maxAlertHistory) {
            alertHistory.pop();
        }

        // Update UI
        renderAlertList();
        updateAlertCount();

        // Play sound for critical alerts
        if (data.priority === 'critical' && CONFIG.alertSoundEnabled && elements.alertSound) {
            elements.alertSound.play().catch(() => {});
        }

        // Show browser notification
        showNotification(data);
    }

    function renderAlertList() {
        if (alertHistory.length === 0) {
            elements.alertList.innerHTML = '<div class="alert-empty">No alerts recorded</div>';
            return;
        }

        const html = alertHistory.slice(0, 10).map(alert => {
            const priorityClass = alert.priority === 'critical' ? 'critical' :
                                  alert.priority === 'warning' ? 'warning' : '';
            const icon = getAlertIcon(alert.type);
            const time = formatAlertTime(alert.timestamp);

            return `
                <div class="alert-item ${priorityClass}" data-id="${alert.id}">
                    <div class="alert-icon">${icon}</div>
                    <div class="alert-content">
                        <div class="alert-message">${escapeHtml(alert.message)}</div>
                        <div class="alert-time">${time}</div>
                    </div>
                </div>
            `;
        }).join('');

        elements.alertList.innerHTML = html;
    }

    function updateAlertCount() {
        elements.alertCount.textContent = alertHistory.length;
    }

    function getAlertIcon(type) {
        switch (type) {
            case 'fall':
                return '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L1 21h22L12 2zm0 3.5L19.5 19h-15L12 5.5z"/></svg>';
            case 'inactivity':
                return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>';
            default:
                return '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 006 8c0 7-3 9-3 9h18s-3-2-3-9"/></svg>';
        }
    }

    function showNotification(alert) {
        if (!('Notification' in window)) return;

        if (Notification.permission === 'granted') {
            new Notification('Fall Detection Alert', {
                body: alert.message,
                icon: '/static/img/icon.png',
                tag: alert.id,
                requireInteraction: alert.priority === 'critical'
            });
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission();
        }
    }

    // ==========================================================================
    // Statistics
    // ==========================================================================

    function fetchStatistics() {
        fetch('/api/statistics')
            .then(response => response.json())
            .then(data => {
                elements.totalFrames.textContent = formatNumber(data.total_frames);
                elements.avgFps.textContent = data.average_fps.toFixed(1);
                elements.avgInference.textContent = data.average_inference_ms.toFixed(1);
                elements.totalFalls.textContent = data.total_falls_detected;
            })
            .catch(error => console.error('Failed to fetch stats:', error));
    }

    function fetchAlertHistory() {
        fetch('/api/alerts')
            .then(response => response.json())
            .then(data => {
                alertHistory = data.reverse();
                renderAlertList();
                updateAlertCount();
            })
            .catch(error => console.error('Failed to fetch alerts:', error));
    }

    // ==========================================================================
    // Actions
    // ==========================================================================

    function sendTestAlert() {
        fetch('/api/test-alert', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Test alert sent:', data);
            })
            .catch(error => console.error('Failed to send test alert:', error));
    }

    function resetDetection() {
        fetch('/api/reset', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('Detection reset:', data);
                hideFallOverlay();
            })
            .catch(error => console.error('Failed to reset:', error));
    }

    function toggleFullscreen() {
        const container = elements.videoContainer;

        if (container.classList.contains('fullscreen')) {
            container.classList.remove('fullscreen');
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        } else {
            container.classList.add('fullscreen');
            if (container.requestFullscreen) {
                container.requestFullscreen();
            }
        }
    }

    // ==========================================================================
    // Utilities
    // ==========================================================================

    function formatUptime(seconds) {
        if (!seconds || seconds < 0) return '--:--:--';

        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);

        return `${pad(hours)}:${pad(minutes)}:${pad(secs)}`;
    }

    function formatActivityState(state) {
        if (!state) return '--';
        return state.charAt(0).toUpperCase() + state.slice(1).replace('_', ' ');
    }

    function formatAlertTime(timestamp) {
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString();
    }

    function formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }

    function pad(num) {
        return num.toString().padStart(2, '0');
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ==========================================================================
    // Event Listeners
    // ==========================================================================

    function setupEventListeners() {
        // Buttons
        elements.testAlertBtn?.addEventListener('click', sendTestAlert);
        elements.resetBtn?.addEventListener('click', resetDetection);
        elements.fullscreenBtn?.addEventListener('click', toggleFullscreen);
        elements.refreshStatsBtn?.addEventListener('click', fetchStatistics);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 't' && !e.ctrlKey && !e.metaKey) {
                sendTestAlert();
            } else if (e.key === 'r' && !e.ctrlKey && !e.metaKey) {
                resetDetection();
            } else if (e.key === 'f' && !e.ctrlKey && !e.metaKey) {
                toggleFullscreen();
            } else if (e.key === 'Escape') {
                elements.videoContainer.classList.remove('fullscreen');
            }
        });

        // Fullscreen change
        document.addEventListener('fullscreenchange', () => {
            if (!document.fullscreenElement) {
                elements.videoContainer.classList.remove('fullscreen');
            }
        });

        // Request notification permission
        if ('Notification' in window && Notification.permission === 'default') {
            setTimeout(() => {
                Notification.requestPermission();
            }, 5000);
        }
    }

    // ==========================================================================
    // Initialization
    // ==========================================================================

    function initialize() {
        console.log('[Dashboard] Initializing...');

        // Initialize WebSocket
        initializeSocket();

        // Setup event listeners
        setupEventListeners();

        // Initial data fetch
        fetchStatistics();
        fetchAlertHistory();

        // Start periodic stats refresh
        statsInterval = setInterval(fetchStatistics, CONFIG.statsRefreshInterval);

        console.log('[Dashboard] Ready');
        console.log('[Dashboard] Keyboard shortcuts: T=Test Alert, R=Reset, F=Fullscreen');
    }

    // Start when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        initialize();
    }

})();
